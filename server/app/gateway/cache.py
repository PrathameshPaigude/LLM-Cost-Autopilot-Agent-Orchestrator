import hashlib
import time
from typing import Optional, Dict, Any

class SemanticCache:
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
        self._store: Dict[str, Dict[str, Any]] = {}

    def _generate_key(self, text: str) -> str:
        normalized = " ".join(text.strip().lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[str]:
        key = self._generate_key(text)
        entry = self._store.get(key)
        if entry:
            if time.time() - entry["timestamp"] < self.ttl_seconds:
                return entry["response"]
            else:
                del self._store[key]
        return None

    def set(self, text: str, response: str):
        key = self._generate_key(text)
        self._store[key] = {
            "response": response,
            "timestamp": time.time()
        }

    def clear(self):
        self._store.clear()

cache = SemanticCache()
