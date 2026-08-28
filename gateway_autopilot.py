import time
import re
import requests
from typing import Dict, Any, Optional

OLLAMA_URL = "http://localhost:11434/api/generate"

# -------------------------------------------------------------
# 1. Feature Extractor & ML Heuristic Classifier
# -------------------------------------------------------------
EDITING_KEYWORDS = re.compile(
    r"\b(correct|grammar|proofread|spelling|rephrase|paraphrase|fix typo|capitalize)\b", 
    re.IGNORECASE
)
CODING_KEYWORDS = re.compile(
    r"\b(write code|implement|train model|algorithm|function|class|pipeline|sql query|build)\b", 
    re.IGNORECASE
)

def evaluate_complexity(prompt: str) -> float:
    """Classifies prompt complexity between 0.0 (Easy) and 1.0 (Hard)."""
    prompt_lower = prompt.lower()
    
    # Priority Branch: Grammar, formatting, proofreading
    if EDITING_KEYWORDS.search(prompt_lower[:60]):
        return 0.12  # Tier 1 (Lightweight)
    
    # Coding & System Architecture
    if CODING_KEYWORDS.search(prompt_lower[:60]):
        if any(term in prompt_lower for term in ["xgboost", "neural", "async", "distributed", "auth", "schema", "architecture"]):
            return 0.85  # Tier 3 (Complex Logic)
        return 0.50  # Tier 2 (Standard Code)
    
    # Word count heuristic for open-ended queries
    words = len(prompt.split())
    if words < 25:
        return 0.20
    elif words < 100:
        return 0.45
    return 0.75

# -------------------------------------------------------------
# 2. Local Ollama Caller
# -------------------------------------------------------------
def call_ollama(model_name: str, prompt: str) -> str:
    """Executes prompt against local Ollama instance (100% Offline)."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            return f"Ollama Error: {response.text}"
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Make sure Ollama is running on localhost:11434."

# -------------------------------------------------------------
# 3. Router & Telemetry Engine
# -------------------------------------------------------------
class CostAutopilotGateway:
    def __init__(self, force_offline_privacy: bool = True):
        self.force_offline_privacy = force_offline_privacy
        self.stats = {
            "total_requests": 0,
            "tier1_calls": 0,
            "tier2_calls": 0,
            "tier3_calls": 0,
            "simulated_dollars_saved": 0.0
        }

    def route_and_execute(self, prompt: str) -> Dict[str, Any]:
        start_time = time.time()
        score = evaluate_complexity(prompt)
        self.stats["total_requests"] += 1

        # Determine Model Target based on Complexity Score
        if score < 0.30:
            # Tier 1: Fast local 1.5B model
            tier = "Tier 1 (Local Qwen 1.5B)"
            model = "qwen2.5:1.5b"
            self.stats["tier1_calls"] += 1
            self.stats["simulated_dollars_saved"] += 0.015

        elif score <= 0.70 or self.force_offline_privacy:
            # Tier 2: Balanced local 8B model (4.8 GB RAM)
            tier = "Tier 2 (Local Llama 3.1 8B)"
            model = "llama3.1:8b-instruct-q4_K_M"
            self.stats["tier2_calls"] += 1
            self.stats["simulated_dollars_saved"] += 0.045

        else:
            # Tier 3 (When online): Cloud Frontier
            tier = "Tier 3 (Local Deep Reasoning Fallback)"
            model = "llama3.1:8b-instruct-q4_K_M"
            self.stats["tier3_calls"] += 1
            self.stats["simulated_dollars_saved"] += 0.120

        # Execute on Local Hardware
        raw_output = call_ollama(model_name=model, prompt=prompt)
        elapsed_sec = round(time.time() - start_time, 2)

        return {
            "prompt": prompt,
            "complexity_score": score,
            "routed_tier": tier,
            "model_used": model,
            "latency_seconds": elapsed_sec,
            "response": raw_output.strip()
        }

    def print_telemetry(self):
        print("\n" + "=" * 65)
        print("         COST AUTOPILOT TELEMETRY & SAVINGS REPORT")
        print("=" * 65)
        print(f" Total Requests Handled:         {self.stats['total_requests']}")
        print(f" ├─ Handled by Tier 1 (Fast):    {self.stats['tier1_calls']}")
        print(f" ├─ Handled by Tier 2 (Mid):     {self.stats['tier2_calls']}")
        print(f" └─ Handled by Tier 3 (Deep):    {self.stats['tier3_calls']}")
        print(f" Actual API Cost Incurred:       $0.00 (100% Free & Offline)")
        print(f" Estimated Cloud Money Saved:    ${self.stats['simulated_dollars_saved']:.4f} USD")
        print("=" * 65 + "\n")


# -------------------------------------------------------------
# 4. Live Test Suite
# -------------------------------------------------------------
if __name__ == "__main__":
    gateway = CostAutopilotGateway(force_offline_privacy=True)

    test_prompts = [
        "Correct the grammar in the sentence: Write a complex code to train xgboost model",
        "Write a Python script to train an XGBoost model with cross-validation and export it to ONNX format",
        "Brainstorm a 3-step offline requirement blueprint for a multi-agent drone telemetry app"
    ]

    for p in test_prompts:
        print(f"\n[Incoming Prompt]: {p}")
        res = gateway.route_and_execute(p)
        print(f" [Router Decision] -> Score: {res['complexity_score']} | Model: {res['model_used']} | Time: {res['latency_seconds']}s")
        print(f" [Preview Response]: {res['response'][:160]}...\n" + "-" * 60)

    gateway.print_telemetry()
