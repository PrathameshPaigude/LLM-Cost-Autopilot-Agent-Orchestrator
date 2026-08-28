import os
import logging
import requests
from typing import Optional
from ..core.config import settings

logger = logging.getLogger(__name__)

class OpenAIProvider:
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                logger.info("openai SDK not installed; falling back to direct REST client.")

    def is_configured(self) -> bool:
        key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "") or self.api_key
        return bool(key and key.strip())

    def generate(self, model: str, prompt: str, system_prompt: Optional[str] = None, timeout: int = 45) -> str:
        api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "") or self.api_key
        if not api_key:
            raise RuntimeError("OpenAI API key is missing. Set OPENAI_API_KEY in .env or environment.")


        # Try official SDK if client initialized
        if self.client:
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                chat = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7
                )
                return chat.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"OpenAI SDK call failed: {e}. Attempting REST fallback...")

        # Resilient Direct REST API Fallback
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
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
                raise RuntimeError(f"OpenAI API Error ({response.status_code}): {response.text}")
        except Exception as e:
            logger.error(f"OpenAI REST invocation failed: {e}")
            raise e

openai_provider = OpenAIProvider()
