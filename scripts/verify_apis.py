import os
import sys

# Ensure parent root is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.app.core.config import settings
from server.app.providers import (
    gemini_provider, 
    groq_provider, 
    openrouter_provider, 
    openai_provider, 
    ollama_provider
)

def test_all_apis():
    print("=" * 65)
    print("        LLM PROVIDER & API KEY VERIFICATION UTILITY")
    print("=" * 65)

    test_prompt = "Say Hello in 3 words."

    providers = [
        ("Google Gemini", gemini_provider, "gemini-2.0-flash"),
        ("Groq Cloud", groq_provider, "llama-3.1-8b-instant"),
        ("OpenRouter (Free)", openrouter_provider, "deepseek/deepseek-r1:free"),
        ("OpenAI", openai_provider, "gpt-4o-mini"),
        ("Local Ollama", ollama_provider, settings.LOCAL_TIER1_MODEL)
    ]

    for name, provider, model in providers:
        print(f"\n[Testing {name}]...")
        
        # Check if API key / local daemon configured
        if hasattr(provider, "is_configured") and not provider.is_configured():
            print("  [-] Status: SKIPPED (API Key Not Configured in .env)")
            continue

        if hasattr(provider, "is_available") and not provider.is_available():
            print("  [-] Status: SKIPPED (Local Ollama daemon not running on localhost:11434)")
            continue

        try:
            response = provider.generate(model=model, prompt=test_prompt, timeout=10)
            preview = (response or "").strip().replace("\n", " ")[:60]
            print("  [OK] Status: ONLINE & WORKING")
            print(f"  + Response Preview: '{preview}...'")
        except Exception as e:
            print(f"  [FAIL] Status: FAILED ({e})")

    print("\n" + "=" * 65 + "\n")

if __name__ == "__main__":
    test_all_apis()
