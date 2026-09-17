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

    def key_for(self, text: str) -> str:
        return self._generate_key(text)

    def set(self, text: str, response: str):
        key = self._generate_key(text)
        self._store[key] = {
            "response": response,
            "timestamp": time.time()
        }

    def clear(self):
        self._store.clear()

    def debug_info(self, text: Optional[str] = None) -> Dict[str, Any]:
        info = {
            "size": len(self._store),
            "ttl_seconds": self.ttl_seconds,
            "key_algorithm": "SHA-256",
            "normalization": "strip, lowercase, collapse whitespace",
            "hit_definition": "normalized prompt hashes to a non-expired stored response",
        }
        if text is not None:
            key = self._generate_key(text)
            entry = self._store.get(key)
            info.update({
                "key": key,
                "normalized_prompt": " ".join(text.strip().lower().split()),
                "hit": bool(entry and time.time() - entry["timestamp"] < self.ttl_seconds),
            })
        return info

cache = SemanticCache()
