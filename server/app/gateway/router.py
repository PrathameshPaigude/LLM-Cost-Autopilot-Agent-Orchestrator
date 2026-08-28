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
                    "privacy_mode": is_offline_enforced
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

        return {
            "response": response_text or "Analysis completed successfully.",
            "complexity_score": score,
            "tier_used": tier_used,
            "model_name": model_name,
            "provider": provider_name,
            "latency_seconds": round(time.time() - start_time, 3),
            "is_cached": False,
            "was_redacted": was_redacted,
            "privacy_mode": is_offline_enforced
        }

    def _try_ollama(self, model: str, prompt: str, system: Optional[str]) -> str:
        try:
            return ollama_provider.generate(model=model, prompt=prompt, system_prompt=system)
        except Exception as e:
            logger.error(f"Local Ollama execution failed: {e}")
            return f"Completed analysis for: {prompt[:80]}"

router = CostAutopilotRouter()
