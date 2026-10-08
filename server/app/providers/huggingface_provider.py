import os
import logging
import requests
from typing import Dict, Optional, Any, List
from ..core.config import settings

logger = logging.getLogger(__name__)

HF_POPULAR_MODELS = {
    # Tier 1 (Lightweight / Fast / Free)
    "meta-llama/Llama-3.1-8B-Instruct": "Tier 1",
    "mistralai/Mistral-7B-Instruct-v0.3": "Tier 1",
    "google/gemma-2-9b-it": "Tier 1",
    "microsoft/Phi-3.5-mini-instruct": "Tier 1",
    
    # Tier 2 (Coding / Balanced Reasoning / Free)
    "Qwen/Qwen2.5-Coder-32B-Instruct": "Tier 2",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "Tier 2",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": "Tier 2",
    
    # Tier 3 (Frontier Architecture / Deep Reasoning / Free)
    "meta-llama/Llama-3.3-70B-Instruct": "Tier 3",
    "Qwen/Qwen2.5-72B-Instruct": "Tier 3"
}

HF_TIER_DEFAULTS = {
    "Tier 1": ["meta-llama/Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3", "google/gemma-2-9b-it"],
    "Tier 2": ["Qwen/Qwen2.5-Coder-32B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"],
    "Tier 3": ["meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen2.5-72B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"]
}

class HuggingFaceProvider:
    def __init__(self):
        pass

    def is_configured(self) -> bool:
        key = settings.HF_API_KEY or os.getenv("HUGGINGFACE_API_KEY", "") or os.getenv("HF_TOKEN", "")
        return bool(key and key.strip())

    def generate(
        self, 
        model: str, 
        prompt: str, 
        system_prompt: Optional[str] = None, 
        timeout: Any = (5.0, 45.0)
    ) -> Dict[str, Any]:
        api_key = settings.HF_API_KEY or os.getenv("HUGGINGFACE_API_KEY", "") or os.getenv("HF_TOKEN", "")
        if not api_key:
            raise RuntimeError("Hugging Face API token (HUGGINGFACE_API_KEY or HF_TOKEN) is missing in .env.")

        target_model = model or settings.HF_DEFAULT_MODEL
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Dual OpenAI-compatible router endpoints for Hugging Face Serverless
        router_endpoints = [
            "https://router.huggingface.co/v1/chat/completions",
            "https://router.huggingface.co/hf-inference/v1/chat/completions"
        ]

        # Candidate models to try in case selected model is temporarily unavailable on serverless
        candidate_models = [target_model]
        # Identify tier of requested model
        tier = HF_POPULAR_MODELS.get(target_model, "Tier 2")
        for alt_model in HF_TIER_DEFAULTS.get(tier, []):
            if alt_model not in candidate_models:
                candidate_models.append(alt_model)

        last_error = None
        for current_model in candidate_models:
            payload = {
                "model": current_model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2048
            }

            for endpoint in router_endpoints:
                try:
                    resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
                    if resp.status_code == 200:
                        data = resp.json()
                        choices = data.get("choices", [])
                        usage = data.get("usage", {})
                        if choices:
                            content = choices[0].get("message", {}).get("content", "").strip()
                            return {
                                "text": content,
                                "input_tokens": usage.get("prompt_tokens"),
                                "output_tokens": usage.get("completion_tokens"),
                                "model_used": current_model
                            }
                    elif resp.status_code == 401:
                        raise RuntimeError(
                            "Hugging Face Authentication Failed (401). Please check that HUGGINGFACE_API_KEY / HF_TOKEN "
                            "in .env is valid and has 'Inference Providers' or Read permissions at https://huggingface.co/settings/tokens"
                        )
                    elif resp.status_code == 402:
                        raise RuntimeError(
                            "Hugging Face Credits Exhausted (402). Your account has no remaining free inference credits on router.huggingface.co. "
                            "You can add credits at https://huggingface.co/settings/billing or continue using Groq, Gemini, and Ollama."
                        )
                    elif resp.status_code == 429:
                        last_error = RuntimeError(f"Hugging Face Rate Limit (429) on model {current_model}.")
                        break  # Rate limit applies to token, try failover
                    elif resp.status_code == 503:
                        last_error = RuntimeError(f"Hugging Face Model Loading/Cold Start (503) for {current_model}.")
                        continue
                    else:
                        last_error = RuntimeError(f"Hugging Face API Error ({resp.status_code}): {resp.text}")
                except requests.RequestException as e:
                    last_error = RuntimeError(f"Hugging Face connection error to {endpoint}: {e}")
                    continue

        if last_error:
            logger.error(f"Hugging Face invocation failed: {last_error}")
            raise last_error

        raise RuntimeError("Hugging Face failed to return a response from all endpoints.")

huggingface_provider = HuggingFaceProvider()
