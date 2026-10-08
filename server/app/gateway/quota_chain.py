"""
Automatic Quota Rollover Chain
==============================
Resilient multi-provider failover engine providing:
1. Automatic quota rollover across free-tier cloud providers
   (Groq -> Gemini -> HuggingFace -> Ollama).
2. Provider health tracking with 429/503 cooldown windows.
3. OpenAI-compatible (/v1/chat/completions) and Anthropic-compatible (/v1/messages)
   proxy endpoints so tools like Claude Code CLI, Cursor, Continue, or Aider
   can execute unlimited free sessions without stopping when a provider's quota runs out.
4. Integrated pre-invocation prompt compression via PromptTokenCompressor.
"""

import time
import uuid
import logging
from typing import Dict, Any, List, Optional, Tuple

from ..core.config import settings
from ..core.prompt_compressor import prompt_compressor
from ..storage.prompt_vault import prompt_vault
from ..providers.gemini_provider import gemini_provider
from ..providers.groq_provider import groq_provider
from ..providers.huggingface_provider import huggingface_provider
from ..providers.ollama_provider import ollama_provider

logger = logging.getLogger(__name__)


class QuotaRolloverChain:
    """
    Automatic Quota Rollover Chain.
    Routes requests through a priority chain of free-tier providers.
    If one provider hits rate limits (429) or goes down (503),
    the request automatically rolls over to the next healthy provider.
    """

    def __init__(self):
        # provider_name -> cooldown_expiry_timestamp
        self._provider_cooldowns: Dict[str, float] = {}
        self._stats = {
            "total_requests": 0,
            "failovers_triggered": 0,
            "quota_skips": 0,
            "total_tokens_compressed": 0,
            "total_tokens_saved": 0
        }

    def _is_cooling_down(self, provider: str) -> bool:
        expiry = self._provider_cooldowns.get(provider, 0.0)
        return time.time() < expiry

    def _mark_cooldown(self, provider: str, cooldown_seconds: float = 60.0):
        self._provider_cooldowns[provider] = time.time() + cooldown_seconds
        logger.warning(f"QuotaChain: Provider {provider} hit rate limit/error. Cooldown for {cooldown_seconds}s.")

    def get_provider_health(self) -> List[Dict[str, Any]]:
        providers = [
            {"name": "Groq Cloud", "configured": groq_provider.is_configured(), "primary_model": settings.GROQ_TIER1_MODEL},
            {"name": "Google Gemini", "configured": gemini_provider.is_configured(), "primary_model": "gemini-flash-latest"},
            {"name": "Hugging Face", "configured": huggingface_provider.is_configured(), "primary_model": "Qwen/Qwen2.5-Coder-32B-Instruct"},
            {"name": "Ollama (Local)", "configured": True, "primary_model": settings.LOCAL_TIER1_MODEL},
        ]
        now = time.time()
        for p in providers:
            expiry = self._provider_cooldowns.get(p["name"], 0.0)
            if expiry > now:
                p["status"] = "cooldown"
                p["cooldown_remaining_seconds"] = round(expiry - now, 1)
            elif p["configured"]:
                p["status"] = "healthy"
                p["cooldown_remaining_seconds"] = 0
            else:
                p["status"] = "unconfigured"
                p["cooldown_remaining_seconds"] = 0
        return providers

    def execute_with_failover(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        compression_mode: str = "balanced",
        preferred_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes a prompt across the failover chain with token compression.
        Automatically shifts to the next provider if the current one 429s or fails.
        Chain order: Groq (fastest) -> Gemini -> HuggingFace -> Ollama (local anchor).
        """
        start_time = time.time()
        self._stats["total_requests"] += 1

        # 1. Compress prompt
        compressed_prompt, comp_metrics = prompt_compressor.compress(prompt, mode=compression_mode)
        self._stats["total_tokens_compressed"] += comp_metrics["compressed_tokens"]
        self._stats["total_tokens_saved"] += comp_metrics["tokens_saved"]

        # 2. Build fast provider priority chain (Groq first for sub-300ms latency)
        chain: List[Tuple[str, Any, str]] = []

        if groq_provider.is_configured():
            chain.append(("Groq Cloud", groq_provider, settings.GROQ_TIER1_MODEL))
        if gemini_provider.is_configured():
            chain.append(("Google Gemini", gemini_provider, "gemini-flash-latest"))
        if huggingface_provider.is_configured():
            chain.append(("Hugging Face", huggingface_provider, "Qwen/Qwen2.5-Coder-32B-Instruct"))
        # Local Ollama is always our ultimate anchor
        chain.append(("Ollama (Local)", ollama_provider, settings.LOCAL_TIER1_MODEL))

        response_text = None
        used_provider = None
        used_model = None
        failovers_occurred = 0
        errors = []

        for provider_name, provider_inst, default_model in chain:
            # Check cooldown
            if self._is_cooling_down(provider_name):
                self._stats["quota_skips"] += 1
                continue

            target_model = default_model
            if preferred_model and preferred_model.lower() not in ("default", "auto", "none", ""):
                if provider_name == "Groq Cloud" and any(k in preferred_model.lower() for k in ("groq", "llama", "qwen", "oss")):
                    target_model = preferred_model
                elif provider_name == "Google Gemini" and "gemini" in preferred_model.lower():
                    target_model = preferred_model
                elif provider_name == "Hugging Face" and any(k in preferred_model.lower() for k in ("huggingface", "qwen", "llama", "mistral", "gemma", "deepseek")):
                    target_model = preferred_model

            try:
                logger.info(f"QuotaChain: Dispatch -> {provider_name} ({target_model})")
                raw_res = provider_inst.generate(target_model, compressed_prompt, system_prompt=system_prompt)
                extracted_text = ""
                if isinstance(raw_res, dict):
                    extracted_text = (raw_res.get("text") or raw_res.get("response") or raw_res.get("content") or "").strip()
                elif isinstance(raw_res, str):
                    extracted_text = raw_res.strip()

                if extracted_text:
                    response_text = extracted_text
                    used_provider = provider_name
                    used_model = getattr(provider_inst, "last_model_used", None) or target_model
                    break
            except Exception as e:
                err_str = str(e)
                logger.warning(f"QuotaChain: {provider_name} failed: {err_str}")
                errors.append(f"{provider_name}: {err_str}")
                failovers_occurred += 1
                self._stats["failovers_triggered"] += 1

                # Lock out on actual rate limit / quota exhaustion
                if "429" in err_str or "quota" in err_str.lower() or "rate limit" in err_str.lower() or "resource_exhausted" in err_str.lower():
                    self._mark_cooldown(provider_name, cooldown_seconds=60.0)
                elif "503" in err_str or "unavailable" in err_str.lower():
                    self._mark_cooldown(provider_name, cooldown_seconds=15.0)
                elif "401" in err_str or "unauthorized" in err_str.lower() or "not found" in err_str.lower():
                    # Broken API key - long cooldown to avoid repeated failures
                    self._mark_cooldown(provider_name, cooldown_seconds=300.0)

        latency_ms = round((time.time() - start_time) * 1000, 1)

        if not response_text:
            response_text = f"All available providers failed. Details: {'; '.join(errors)}"
            used_provider = "None"
            used_model = "None"

        # Record in PromptVault SQLite
        tokens_orig = comp_metrics["original_tokens"]
        tokens_comp = comp_metrics["compressed_tokens"]
        tokens_out = max(1, len(response_text.split()) * 4 // 3)

        vault_id = prompt_vault.store_prompt(
            original_prompt=prompt,
            compressed_prompt=compressed_prompt,
            tokens_original=tokens_orig,
            tokens_compressed=tokens_comp,
            compression_ratio=comp_metrics["compression_ratio"],
            complexity_score=0.5,
            tier_used="Quota Rollover",
            provider=used_provider or "Unknown",
            model_name=used_model or "Unknown",
            response_snippet=response_text[:300],
            tokens_in=tokens_comp,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            agent_name="QuotaChain"
        )

        return {
            "response": response_text,
            "provider": used_provider,
            "model": used_model,
            "failovers_count": failovers_occurred,
            "latency_ms": latency_ms,
            "compression": comp_metrics,
            "vault_id": vault_id
        }


quota_chain = QuotaRolloverChain()
