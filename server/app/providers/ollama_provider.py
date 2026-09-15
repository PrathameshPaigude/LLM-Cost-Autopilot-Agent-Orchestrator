import requests
import logging
from typing import Optional, List
from ..core.config import settings

logger = logging.getLogger(__name__)

class OllamaProvider:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def list_installed_models(self) -> List[str]:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if r.status_code == 200:
                data = r.json()
                return [m.get("name") for m in data.get("models", [])]
            return []
        except Exception:
            return []

    def generate(self, model: str, prompt: str, system_prompt: Optional[str] = None, timeout: int = 180) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7
            }
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(url, json=payload, timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                result = data.get("response", "").strip()
                if not result:
                    raise RuntimeError("Ollama returned an empty response.")
                return result
            else:
                logger.error(f"Ollama returned error {response.status_code}: {response.text}")
                raise RuntimeError(f"Ollama error: {response.text}")
        except requests.exceptions.ConnectionError:
            logger.error("Could not connect to Ollama. Ensure Ollama is running on localhost:11434.")
            raise RuntimeError("Ollama connection failed. Is Ollama running?")
        except Exception as e:
            logger.error(f"Ollama unexpected failure: {e}")
            raise e

ollama_provider = OllamaProvider()
