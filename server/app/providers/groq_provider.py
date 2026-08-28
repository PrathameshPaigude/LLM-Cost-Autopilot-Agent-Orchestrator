import os
import time
import logging
import requests
from typing import Optional, List
from ..core.config import settings

logger = logging.getLogger(__name__)

GROQ_FALLBACK_MODELS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "groq/compound-mini",
    "openai/gpt-oss-120b"
]

class GroqProvider:
    def __init__(self):
        self.api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")

    def is_configured(self) -> bool:
        key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "") or self.api_key
        return bool(key and key.strip() and not key.startswith("sk-or-"))

    def generate(self, model: str, prompt: str, system_prompt: Optional[str] = None, timeout: int = 15) -> str:
        api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "") or self.api_key
        if not api_key:
            raise RuntimeError("Groq API key is missing.")

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Try primary model and pool of fallback models if rate-limited
        models_to_try = [model] + [m for m in GROQ_FALLBACK_MODELS if m != model]

        last_error = None
        for m in models_to_try:
            payload = {
                "model": m,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1024
            }
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=timeout)
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"].strip()
                elif response.status_code == 429:
                    logger.warning(f"Groq model {m} rate limited (429). Trying alternate pool model...")
                    time.sleep(1.0)
                    continue
                else:
                    last_error = f"Groq error ({response.status_code}): {response.text}"
            except Exception as e:
                last_error = str(e)
                continue

        raise RuntimeError(last_error or "All Groq model pools exhausted.")

groq_provider = GroqProvider()
