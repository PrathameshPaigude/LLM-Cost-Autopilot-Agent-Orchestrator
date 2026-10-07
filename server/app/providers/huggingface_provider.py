import os
import logging
import requests
from typing import Dict, Optional, Any
from ..core.config import settings

logger = logging.getLogger(__name__)

HF_POPULAR_MODELS = {
    "Qwen/Qwen2.5-Coder-32B-Instruct": "Tier 2",
    "meta-llama/Llama-3.1-8B-Instruct": "Tier 1",
    "mistralai/Mistral-7B-Instruct-v0.3": "Tier 1",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": "Tier 2",
    "bigcode/starcoder2-15b": "Tier 2",
    "microsoft/Phi-3.5-mini-instruct": "Tier 1",
    "google/gemma-2-9b-it": "Tier 1"
}

class HuggingFaceProvider:
    def __init__(self):
        self.api_key = settings.HF_API_KEY

    def is_configured(self) -> bool:
        key = settings.HF_API_KEY or os.getenv("HUGGINGFACE_API_KEY", "") or os.getenv("HF_TOKEN", "")
        return bool(key and key.strip())

    def generate(
        self, 
        model: str, 
        prompt: str, 
        system_prompt: Optional[str] = None, 
        timeout: Any = (5.0, 30.0)
    ) -> Dict[str, Any]:
        api_key = settings.HF_API_KEY or os.getenv("HUGGINGFACE_API_KEY", "") or os.getenv("HF_TOKEN", "")
        if not api_key:
            raise RuntimeError("Hugging Face API token (HF_TOKEN or HUGGINGFACE_API_KEY) is missing.")

        target_model = model or settings.HF_DEFAULT_MODEL
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Method 1: Try OpenAI-compatible HF Router endpoint
        router_url = "https://router.huggingface.co/hf-inference/v1/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        router_payload = {
            "model": target_model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048
        }

        try:
            resp = requests.post(router_url, headers=headers, json=router_payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                usage = data.get("usage", {})
                if choices:
                    content = choices[0].get("message", {}).get("content", "").strip()
                    return {
                        "text": content,
                        "input_tokens": usage.get("prompt_tokens"),
                        "output_tokens": usage.get("completion_tokens")
                    }
        except Exception as e:
            logger.debug(f"HF Router endpoint failed ({e}), attempting standard model inference API...")

        # Method 2: Direct model endpoint fallback
        direct_url = f"https://api-inference.huggingface.co/models/{target_model}"
        full_input = f"{system_prompt}\n\nUser: {prompt}" if system_prompt else prompt
        direct_payload = {
            "inputs": full_input,
            "parameters": {
                "max_new_tokens": 1024,
                "temperature": 0.7,
                "return_full_text": False
            }
        }

        try:
            resp = requests.post(direct_url, headers=headers, json=direct_payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    generated_text = data[0].get("generated_text", "")
                elif isinstance(data, dict):
                    generated_text = data.get("generated_text", str(data))
                else:
                    generated_text = str(data)

                input_est = max(1, len(full_input.split()) * 4 // 3)
                output_est = max(1, len(generated_text.split()) * 4 // 3)
                return {
                    "text": generated_text.strip(),
                    "input_tokens": input_est,
                    "output_tokens": output_est
                }
            elif resp.status_code == 503:
                # Model is loading
                err = resp.json().get("error", "Model currently loading on Hugging Face Serverless.")
                raise RuntimeError(f"Hugging Face Model Loading: {err}")
            else:
                raise RuntimeError(f"Hugging Face API Error ({resp.status_code}): {resp.text}")
        except Exception as e:
            logger.error(f"Hugging Face inference failed: {e}")
            raise e

huggingface_provider = HuggingFaceProvider()
