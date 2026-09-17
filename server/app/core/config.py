import os
from typing import Optional
from dotenv import load_dotenv

def reload_env():
    if os.path.exists(".env"):
        load_dotenv(".env", override=True)
    elif os.path.exists(".env.example"):
        load_dotenv(".env.example", override=True)

# Initial load
reload_env()

try:
    from pydantic_settings import BaseSettings
    class BaseConfig(BaseSettings):
        class Config:
            env_file = (".env", ".env.example")
            extra = "allow"
except ImportError:
    from pydantic import BaseModel
    class BaseConfig(BaseModel):
        pass

class Settings(BaseConfig):
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Agent Orchestration Platform")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1")

    # Privacy & Offline Mode
    PRIVACY_MODE: bool = os.getenv("PRIVACY_MODE", "False").lower() in ("true", "1")

    # Local Ollama Settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LOCAL_TIER1_MODEL: str = os.getenv("LOCAL_TIER1_MODEL", "qwen2.5:1.5b")
    LOCAL_TIER2_MODEL: str = os.getenv("LOCAL_TIER2_MODEL", "llama3.1:8b-instruct-q4_K_M")
    LOCAL_TIER1_TIMEOUT_SECONDS: int = int(os.getenv("LOCAL_TIER1_TIMEOUT_SECONDS", "60"))
    LOCAL_TIER2_TIMEOUT_SECONDS: int = int(os.getenv("LOCAL_TIER2_TIMEOUT_SECONDS", "180"))
    LOCAL_TIER3_TIMEOUT_SECONDS: int = int(os.getenv("LOCAL_TIER3_TIMEOUT_SECONDS", "300"))

    # Dynamic API Key Properties
    @property
    def GEMINI_API_KEY(self) -> str:
        reload_env()
        return os.getenv("GEMINI_API_KEY", "")

    @property
    def GROQ_API_KEY(self) -> str:
        reload_env()
        return os.getenv("GROQ_API_KEY", "")

    @property
    def OPENAI_API_KEY(self) -> str:
        reload_env()
        return os.getenv("OPENAI_API_KEY", "")

    @property
    def OPENROUTER_API_KEY(self) -> str:
        reload_env()
        return os.getenv("OPENROUTER_API_KEY", "")

    # Routing Thresholds
    ROUTING_TIER1_MAX: float = float(os.getenv("ROUTING_TIER1_MAX", "0.30"))
    ROUTING_TIER2_MAX: float = float(os.getenv("ROUTING_TIER2_MAX", "0.70"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.80"))

settings = Settings()
