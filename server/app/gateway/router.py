import time
import logging
from typing import Dict, Any, Optional

from ..core.config import settings
from ..core.telemetry import telemetry
from .classifier import classifier
from .cache import cache
from .redactor import redactor
from ..providers.ollama_provider import ollama_provider
from ..providers.groq_provider import groq_provider
from ..providers.gemini_provider import gemini_provider
from ..providers.openai_provider import openai_provider
from ..providers.openrouter_provider import openrouter_provider

logger = logging.getLogger(__name__)

class CostAutopilotRouter:
    MODEL_COSTS = {
        "Claude-class baseline": {"input": 15.0, "output": 75.0, "confidence": 0.94},
        "GPT-4o-class baseline": {"input": 5.0, "output": 15.0, "confidence": 0.90},
        "Gemini Flash": {"input": 0.075, "output": 0.30, "confidence": 0.82},
        "Groq fast tier": {"input": 0.0, "output": 0.0, "confidence": 0.78},
        "Ollama local": {"input": 0.0, "output": 0.0, "confidence": 0.70},
    }

    def route_and_execute(
        self, 
        prompt: str, 
        agent_name: str = "General", 
        system_prompt: Optional[str] = None,
        force_offline: Optional[bool] = None,
        provider_override: Optional[str] = None,
        model_override: Optional[str] = None
    ) -> Dict[str, Any]:
        start_time = time.time()
        is_offline_enforced = settings.PRIVACY_MODE if force_offline is None else force_offline

        # 1. PII Redaction
        sanitized_prompt, was_redacted = redactor.sanitize(prompt)

        # 2. Semantic Cache Check (Tier 0)
        if not provider_override and not model_override:
            cached_result = cache.get(sanitized_prompt)
            if cached_result:
                audit = self._build_audit(
                    sanitized_prompt, agent_name, 0.0, "Tier 0 (Semantic Cache)", "Cache", "cache", 0, 0,
                    is_offline_enforced, was_redacted, is_cached=True
                )
                telemetry.record_request("Tier 0 (Semantic Cache)", sanitized_prompt, cached_result, is_cache=True)
                return {
                    "response": cached_result,
                    "complexity_score": 0.0,
                    "tier_used": "Tier 0 (Semantic Cache)",
                    "model_name": "cache",
                    "provider": "Cache",
                    "latency_seconds": round(time.time() - start_time, 4),
                    "is_cached": True,
                    "was_redacted": was_redacted,
                    "privacy_mode": is_offline_enforced,
                    "routing_audit": audit
                }

        # 3. Complexity Classification (0.0 to 1.0)
        score = classifier.predict_score(sanitized_prompt, agent_name=agent_name)

        # 4. Model Selection & Execution
        tier_used = ""
        model_name = ""
        provider_name = ""
        response_text = ""

        # Priority 1: Handle Manual Provider Override if requested
        if provider_override:
            prov = provider_override.lower()
            try:
                if prov == "groq" and groq_provider.is_configured():
                    model_name = model_override or "openai/gpt-oss-20b"
                    provider_name = "Groq Cloud"
                    tier_used = "Manual (Groq Cloud)"
                    response_text = groq_provider.generate(model_name, sanitized_prompt, system_prompt)
                elif prov == "gemini" and gemini_provider.is_configured():
                    model_name = model_override or "gemini-flash-latest"
                    provider_name = "Google Gemini"
                    tier_used = "Manual (Google Gemini)"
                    response_text = gemini_provider.generate(model_name, sanitized_prompt, system_prompt)
                elif prov == "openai" and openai_provider.is_configured():
                    model_name = model_override or "gpt-4o-mini"
                    provider_name = "OpenAI"
                    tier_used = "Manual (OpenAI)"
                    response_text = openai_provider.generate(model_name, sanitized_prompt, system_prompt)
                elif prov == "openrouter" and openrouter_provider.is_configured():
                    model_name = model_override or "deepseek/deepseek-r1:free"
                    provider_name = "OpenRouter"
                    tier_used = "Manual (OpenRouter)"
                    response_text = openrouter_provider.generate(model_name, sanitized_prompt, system_prompt)
                elif prov == "ollama":
                    model_name = model_override or settings.LOCAL_TIER2_MODEL
                    provider_name = "Ollama (Local)"
                    tier_used = "Manual (Local Ollama)"
                    response_text = ollama_provider.generate(model_name, sanitized_prompt, system_prompt)
            except Exception as e:
                logger.warning(f"Manual override provider {prov} failed: {e}. Falling back to dynamic matrix...")

        # Priority 2: Automatic 3-Tier Execution Matrix
        if not response_text:
            if is_offline_enforced:
                # 100% Offline Local Ollama
                target_model = settings.LOCAL_TIER1_MODEL if score < settings.ROUTING_TIER1_MAX else settings.LOCAL_TIER2_MODEL
                tier_used = "Tier 1 (Local Qwen 1.5B)" if score < settings.ROUTING_TIER1_MAX else "Tier 2/3 (Local Llama 3.1 8B)"
                provider_name = "Ollama (Local)"
                model_name = target_model
                response_text = self._try_ollama(target_model, sanitized_prompt, system_prompt)

            else:
                # Online Dynamic Routing: Groq (ultra fast) -> Gemini Flash -> Local Ollama
                if score < settings.ROUTING_TIER1_MAX:
                    tier_used = "Tier 1 (Fast Lightweight)"
                    model_name = "openai/gpt-oss-20b"
                elif score <= settings.ROUTING_TIER2_MAX:
                    tier_used = "Tier 2 (Balanced Reasoning)"
                    model_name = "openai/gpt-oss-20b"
                else:
                    tier_used = "Tier 3 (Frontier Reasoning)"
                    model_name = "openai/gpt-oss-20b"

                # Step A: Try Groq Cloud
                if groq_provider.is_configured():
                    try:
                        provider_name = "Groq Cloud"
                        response_text = groq_provider.generate(model_name, sanitized_prompt, system_prompt)
                    except Exception as e:
                        logger.warning(f"Groq failed: {e}")

                # Step B: Fallback to Gemini Flash
                if not response_text and gemini_provider.is_configured():
                    try:
                        provider_name = "Google Gemini"
                        model_name = "gemini-flash-latest"
                        response_text = gemini_provider.generate(model_name, sanitized_prompt, system_prompt)
                    except Exception as e:
                        logger.warning(f"Gemini fallback failed: {e}")

                # Step C: Fallback to Local Ollama
                if not response_text:
                    provider_name = "Ollama (Local Fallback)"
                    local_model = settings.LOCAL_TIER1_MODEL if score < settings.ROUTING_TIER1_MAX else settings.LOCAL_TIER2_MODEL
                    model_name = local_model
                    response_text = self._try_ollama(local_model, sanitized_prompt, system_prompt)

        # 5. Populate Cache & Telemetry
        if response_text:
            cache.set(sanitized_prompt, response_text)
            telemetry.record_request(tier_used, sanitized_prompt, response_text, is_cache=False)

        input_tokens = max(1, len(sanitized_prompt.split()) * 4 // 3)
        output_tokens = max(1, len(response_text.split()) * 4 // 3)
        audit = self._build_audit(
            sanitized_prompt, agent_name, score, tier_used, provider_name, model_name,
            input_tokens, output_tokens, is_offline_enforced, was_redacted,
            is_cached=False, manual_override=bool(provider_override or model_override)
        )

        return {
            "response": response_text or "",
            "error": "No provider returned a response." if not response_text else None,
            "complexity_score": score,
            "tier_used": tier_used,
            "model_name": model_name,
            "provider": provider_name,
            "latency_seconds": round(time.time() - start_time, 3),
            "is_cached": False,
            "was_redacted": was_redacted,
            "privacy_mode": is_offline_enforced,
            "routing_audit": audit
        }

    def _build_audit(
        self, prompt: str, agent_name: str, score: float, tier: str, provider: str,
        model: str, input_tokens: int, output_tokens: int, offline: bool,
        redacted: bool, is_cached: bool, manual_override: bool = False
    ) -> Dict[str, Any]:
        features = classifier.extract_features(prompt, agent_name=agent_name)
        estimated_baseline = self._estimate_cost("GPT-4o-class baseline", input_tokens, output_tokens)
        selected_label = self._cost_label(provider, offline, is_cached)
        selected_cost = self._estimate_cost(selected_label, input_tokens, output_tokens)
        alternatives = []
        for label, rates in self.MODEL_COSTS.items():
            cost = self._estimate_cost(label, input_tokens, output_tokens)
            alternatives.append({
                "option": label,
                "estimated_cost_usd": round(cost, 6),
                "estimated_confidence": rates["confidence"],
                "relative_savings_vs_claude_usd": round(max(0.0, estimated_baseline - cost), 6),
            })
        return {
            "decision": "cache_hit" if is_cached else ("manual_override" if manual_override else "automatic"),
            "reason": self._reason(score, features, offline, is_cached, manual_override),
            "complexity": {
                "score": score,
                "tier": tier,
                "features": features,
            },
            "selected": {
                "provider": provider,
                "model": model,
                "estimated_cost_usd": round(selected_cost, 6),
                "estimated_confidence": self.MODEL_COSTS[selected_label]["confidence"],
            },
            "alternatives": alternatives,
            "estimated_gpt4_class_cost_usd": round(estimated_baseline, 6),
            "estimated_savings_vs_gpt4_class_usd": round(max(0.0, estimated_baseline - selected_cost), 6),
            "cost_basis": "Estimated using token counts and illustrative provider rates; not a billing record.",
            "privacy_mode": offline,
            "was_redacted": redacted,
        }

    def _cost_label(self, provider: str, offline: bool, is_cached: bool) -> str:
        if is_cached:
            return "Ollama local"
        if offline or "Ollama" in provider:
            return "Ollama local"
        if "Gemini" in provider:
            return "Gemini Flash"
        if "Groq" in provider or "OpenRouter" in provider:
            return "Groq fast tier"
        return "GPT-4o-class baseline" if "OpenAI" in provider else "Ollama local"

    def _estimate_cost(self, label: str, input_tokens: int, output_tokens: int) -> float:
        rates = self.MODEL_COSTS[label]
        return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000

    @staticmethod
    def _reason(score: float, features: Dict[str, Any], offline: bool, cached: bool, manual: bool) -> str:
        if cached:
            return "Exact cached response avoided a new model call."
        if manual:
            return "A provider or model override was explicitly selected by the user."
        if offline:
            return "Privacy mode forced local execution regardless of complexity."
        if features["is_editing"]:
            return "Editing intent reduced complexity despite any technical terms in the payload."
        if features["has_deep_domain"]:
            return "Advanced-domain signals increased the complexity score and favored deeper reasoning."
        return f"Complexity score {score:.2f} matched the configured cost-aware execution tier."

    def _try_ollama(self, model: str, prompt: str, system: Optional[str]) -> str:
        try:
            return ollama_provider.generate(model=model, prompt=prompt, system_prompt=system)
        except Exception as e:
            logger.error(f"Local Ollama execution failed: {e}")
            return ""

router = CostAutopilotRouter()
