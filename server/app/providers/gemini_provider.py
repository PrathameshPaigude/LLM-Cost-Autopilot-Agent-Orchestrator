import os
import logging
import requests
from typing import Optional, Any
from ..core.config import settings


logger = logging.getLogger(__name__)

GEMINI_MODEL_MAP = {
    "gemini-2.0-flash": "gemini-flash-latest",
    "gemini-2.0-flash-lite": "gemini-flash-lite-latest",
    "gemini-1.5-flash": "gemini-flash-latest",
    "gemini-1.5-pro": "gemini-pro-latest",
    "gemini-2.5-pro": "gemini-pro-latest"
}

class GeminiProvider:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")

    def is_configured(self) -> bool:
        key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "") or self.api_key
        return bool(key and key.strip())

    def generate(self, model: str, prompt: str, system_prompt: Optional[str] = None, timeout: Any = (3.0, 8.0)) -> str:
        api_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "") or self.api_key
        if not api_key:
            raise RuntimeError("Gemini API key is missing.")


        target_model = GEMINI_MODEL_MAP.get(model, model)
        if target_model.startswith("models/"):
            target_model = target_model[len("models/"):]

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        
        contents_payload = []
        if system_prompt:
            contents_payload.append({
                "role": "user",
                "parts": [{"text": f"[System Context]: {system_prompt}"}]
            })
            contents_payload.append({
                "role": "model",
                "parts": [{"text": "Understood. I will follow these instructions."}]
            })
            
        contents_payload.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })

        payload = {
            "contents": contents_payload,
            "generationConfig": {
                "temperature": 0.7
            }
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                candidates = data.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    parts = candidates[0]["content"].get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip()
                return "Gemini returned empty response content."
            else:
                raise RuntimeError(f"Gemini REST API Error ({response.status_code}): {response.text}")
        except Exception as e:
            logger.error(f"Gemini REST invocation failed: {e}")
            raise e

gemini_provider = GeminiProvider()
