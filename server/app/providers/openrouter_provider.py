import os
import logging
import requests
from typing import Optional
from ..core.config import settings

logger = logging.getLogger(__name__)

class OpenRouterProvider:
    def __init__(self):
        self.api_key = settings.OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY", "")

    def is_configured(self) -> bool:
        key = settings.OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY", "") or self.api_key
        return bool(key and key.strip())

    def generate(self, model: str, prompt: str, system_prompt: Optional[str] = None, timeout: int = 45) -> str:
        api_key = settings.OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY", "") or self.api_key
        if not api_key:
            raise RuntimeError("OpenRouter API key is missing. Set OPENROUTER_API_KEY in .env or environment.")


        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Agent-Orchestrator",
            "Content-Type": "application/json"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                raise RuntimeError(f"OpenRouter API Error ({response.status_code}): {response.text}")
        except Exception as e:
            logger.error(f"OpenRouter invocation failed: {e}")
            raise e

openrouter_provider = OpenRouterProvider()
