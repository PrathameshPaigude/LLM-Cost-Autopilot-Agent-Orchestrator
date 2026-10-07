"""
Production ML Classifier Training Pipeline: Prompt Complexity & Tier Router
=============================================================================
Ingests:
  1. Synthetic template dataset (bootstrapping base)
  2. HuggingFace routing benchmarks (R2-Bench / RoutingCompendium from data/benchmark_prompts.jsonl)
  3. Passively accumulated live routing observations (from data/routing_observations.jsonl)

Trains Gradient Boosted Trees (XGBoost / GBDT) on TF-IDF + linguistic features
to classify prompt complexity into:
  - Tier 1 (0.00 - 0.30) -> Fast lightweight models (Groq / Gemini Flash Lite / Ollama 1.5B)
  - Tier 2 (0.31 - 0.70) -> Balanced reasoning & code (Gemini 2.0 Flash / Qwen 2.5 Coder 32B / Ollama 8B)
  - Tier 3 (0.71 - 1.00) -> Frontier reasoning & architecture (DeepSeek R1 / OpenAI / Claude)

Usage:
  python scripts/train_router_model.py
  python scripts/train_router_model.py --include-benchmarks
  python scripts/train_router_model.py --include-observations --min-obs 10
"""

import os
import re
import json
import joblib
import time
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGBOOST = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
GATEWAY_DIR = PROJECT_ROOT / "server" / "app" / "gateway"

TIER_1_TEMPLATES = [
    "Correct the grammar in the sentence: {payload}",
    "Fix any typos in: {payload}",
    "Proofread this short note: {payload}",
    "Capitalize all words in: {payload}",
    "Format this json payload cleanly: {payload}",
    "Rephrase this email to sound more professional: {payload}",
    "Translate this sentence to Spanish: {payload}",
    "Remove duplicate lines from this list: {payload}",
    "Convert this text to uppercase: {payload}",
    "Spellcheck the following paragraph: {payload}",
    "What is the definition of {payload}?",
    "Summarize this 1-line update: {payload}",
    "Make this text more concise: {payload}",
    "Change tone to friendly: {payload}",
    "Format this phone number: {payload}",
    "Check syntax of this simple snippet: {payload}",
    "Generate a bullet list of: {payload}",
    "Extract emails from: {payload}",
]

TIER_2_TEMPLATES = [
    "Write a Python script to {payload}",
    "Implement a FastAPI REST endpoint that handles {payload}",
    "Write a SQL query to {payload}",
    "Create a React component with state to {payload}",
    "Write unit tests using pytest for {payload}",
    "Build a data parsing pipeline in pandas for {payload}",
    "Create a Dockerfile and docker-compose setup for {payload}",
    "Explain how to configure nginx for {payload}",
    "Write a bash script to automate {payload}",
    "Implement a Redis caching layer for {payload}",
    "Compare the performance differences between {payload}",
    "Refactor this Python function to improve readability: {payload}",
    "Write a Node.js utility function to {payload}",
    "Create a GitHub Action workflow to {payload}",
    "Implement rate limiting middleware for {payload}",
    "Design a SQLite schema and migration script for {payload}",
]

TIER_3_TEMPLATES = [
    "Design a distributed, fault-tolerant architecture for {payload} handling 100k TPS with CAP theorem tradeoffs",
    "Implement a lock-free concurrent data structure in C++ for {payload} with atomic memory ordering",
    "Derive the mathematical proof for {payload} from first principles",
    "Architect an end-to-end multi-region active-active database replication system with Raft consensus for {payload}",
    "Implement full Multi-Head Self Attention and FlashAttention in PyTorch from scratch for {payload}",
    "Design a zero-knowledge proof (zk-SNARK) verification protocol for {payload}",
    "Analyze the asymptotic time and space complexity of {payload} under non-uniform distribution",
    "Develop a real-time kernel memory tracing tool using eBPF and Linux tracepoints for {payload}",
    "Write a complete Deep Reinforcement Learning pipeline using PPO and Actor-Critic for {payload}",
    "Design a sharded consistent hashing vector database storage engine for {payload} with 10B items",
    "Architect a real-time YOLOv8 edge computer vision obstacle avoidance pipeline with TensorRT and Kalman filtering for {payload}",
    "Derive the convergence guarantees for stochastic gradient descent on non-convex Riemannian manifolds for {payload}",
    "Design an end-to-end BGP routing policy engine with fault recovery and zero packet drop guarantees for {payload}",
]

PAYLOAD_SAMPLES = [
    "user registration and password hashing",
    "calculating customer monthly recurring revenue and churn",
    "fetching and parsing weather data from external API",
    "sorting a list of timestamps in ascending order",
    "validating email addresses with regular expressions",
    "managing shopping cart state with local storage",
    "optimizing database connection pooling",
    "implementing JWT token authentication with refresh tokens",
    "handling WebSocket connections for real-time notifications",
    "distributed event streaming with Kafka partition rebalancing",
    "autonomous drone navigation in GPS-denied environments",
    "cryptographic key exchange using elliptic curve Diffie-Hellman",
    "gradient descent convergence on non-convex manifolds",
    "high-frequency algorithmic order book matching engine",
    "microservices telemetry ingestion with distributed tracing",
    "multi-tenant role-based access control with hierarchical permissions",
]


def generate_synthetic_dataset() -> Tuple[List[str], List[int]]:
    prompts = []
    labels = []

    for t in TIER_1_TEMPLATES:
        for p in PAYLOAD_SAMPLES:
            prompts.append(t.format(payload=p))
            labels.append(0)

    for t in TIER_2_TEMPLATES:
        for p in PAYLOAD_SAMPLES:
            prompts.append(t.format(payload=p))
            labels.append(1)

    for t in TIER_3_TEMPLATES:
        for p in PAYLOAD_SAMPLES:
            prompts.append(t.format(payload=p))
            labels.append(2)

    return prompts, labels


def load_benchmark_prompts(max_rows: int = 5000) -> Tuple[List[str], List[int]]:
    """Loads benchmark prompts downloaded from HuggingFace (R2-Bench / RoutingCompendium)."""
    prompts = []
    labels = []
    bench_file = DATA_DIR / "benchmark_prompts.jsonl"
    if not bench_file.exists():
        return prompts, labels

    try:
        with open(bench_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    prompt = entry.get("prompt")
                    tier = entry.get("tier")
                    if prompt and tier in (0, 1, 2):
                        prompts.append(prompt)
                        labels.append(tier)
                        if len(prompts) >= max_rows:
                            break
                except (json.JSONDecodeError, KeyError):
                    continue
    except Exception as e:
        print(f"Warning: Failed to load benchmark prompts: {e}")

    return prompts, labels


def load_live_observations(min_samples: int = 10) -> Tuple[List[str], List[int]]:
    """Loads passively recorded observations from quality_tracker."""
    prompts = []
    labels = []
    obs_file = DATA_DIR / "routing_observations.jsonl"
    if not obs_file.exists():
        return prompts, labels

    tier_map = {
        "Tier 1": 0,
        "Tier 2": 1,
        "Tier 3": 2,
    }

    try:
        with open(obs_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    prompt = entry.get("prompt_snippet") or entry.get("prompt")
                    tier_str = entry.get("tier", "")
                    if not prompt or len(prompt.strip()) < 5:
                        continue

                    # Map tier string to int
                    assigned_tier = None
                    for k, v in tier_map.items():
                        if k in tier_str:
                            assigned_tier = v
                            break

                    if assigned_tier is not None:
                        prompts.append(prompt)
                        labels.append(assigned_tier)
                except (json.JSONDecodeError, KeyError):
                    continue
    except Exception as e:
        print(f"Warning: Failed to load live observations: {e}")

    if len(prompts) < min_samples:
        print(f"Live observations ({len(prompts)}) below threshold ({min_samples}); skipped.")
        return [], []

    return prompts, labels


class FeatureExtractor:
    """
    Extracts 6 linguistic & structural features matching server/app/gateway/classifier.py.
    """
    def __init__(self):
        self.editing_re = re.compile(
            r"\b(correct|grammar|proofread|spelling|rephrase|fix typo|capitalize|format json|clean text)\b",
            re.IGNORECASE
        )
        self.coding_re = re.compile(
            r"\b(write code|implement|build|develop|create function|pipeline|script|sql query|fastapi|react|endpoint)\b",
            re.IGNORECASE
        )
        self.frontier_re = re.compile(
            r"\b(distributed|fault[- ]tolerant|lock[- ]free|raft|consensus|ebpf|zero[- ]knowledge|reinforcement learning|kalman|sharded|100k tps|proof|derive)\b",
            re.IGNORECASE
        )

    def extract(self, texts: List[str]) -> np.ndarray:
        features = []
        for t in texts:
            t_str = str(t)
            t_low = t_str.lower()
            prefix = t_low[:80]

            is_editing = 1.0 if self.editing_re.search(prefix) else 0.0
            is_coding = 1.0 if (self.coding_re.search(prefix) and not is_editing) else 0.0
            is_frontier = 1.0 if self.frontier_re.search(t_low) else 0.0
            has_quotes = 1.0 if ('"' in t_str or "'" in t_str or "`" in t_str) else 0.0
            words = len(t_str.split())

            features.append([
                is_editing,
                is_coding,
                is_frontier,
                has_quotes,
                min(words / 150.0, 1.0),
                min(len(t_str) / 800.0, 1.0)
            ])
        return np.array(features, dtype=np.float32)


def train_and_export(
    export_dir: Optional[str] = None,
    include_benchmarks: bool = True,
    include_observations: bool = True,
    min_obs: int = 10
) -> Dict[str, Any]:
    export_path = Path(export_dir) if export_dir else GATEWAY_DIR
    export_path.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Agent Orchestrator - Router Classifier Training Pipeline")
    print("=" * 65)

    # 1. Dataset assembly
    X_synthetic, y_synthetic = generate_synthetic_dataset()
    print(f"Synthetic template samples : {len(X_synthetic)}")

    X_all = list(X_synthetic)
    y_all = list(y_synthetic)
    bench_count = 0
    obs_count = 0

    if include_benchmarks:
        X_bench, y_bench = load_benchmark_prompts()
        if X_bench:
            bench_count = len(X_bench)
            X_all.extend(X_bench)
            y_all.extend(y_bench)
            print(f"HF Benchmark samples loaded: {bench_count}")

    if include_observations:
        X_obs, y_obs = load_live_observations(min_samples=min_obs)
        if X_obs:
            obs_count = len(X_obs)
            X_all.extend(X_obs)
            y_all.extend(y_obs)
            print(f"Live observation samples   : {obs_count}")

    total_samples = len(X_all)
    print(f"Total training corpus      : {total_samples} samples across 3 tiers")

    # Tier counts
    tier_counts = {0: y_all.count(0), 1: y_all.count(1), 2: y_all.count(2)}
    print(f"Tier distribution: Tier 1={tier_counts[0]}, Tier 2={tier_counts[1]}, Tier 3={tier_counts[2]}")

    # Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    # Feature extraction: 800 TF-IDF + 6 linguistic = 806 features
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=800, sublinear_tf=True)
    extractor = FeatureExtractor()

    X_train_tfidf = tfidf.fit_transform(X_train_raw).toarray()
    X_train_ling = extractor.extract(X_train_raw)
    X_train = np.hstack([X_train_tfidf, X_train_ling])

    X_test_tfidf = tfidf.transform(X_test_raw).toarray()
    X_test_ling = extractor.extract(X_test_raw)
    X_test = np.hstack([X_test_tfidf, X_test_ling])

    print(f"Engineered feature shape   : {X_train.shape[1]} (800 TF-IDF + 6 linguistic)")

    # Model training
    print("\nTraining classifier...")
    if HAS_XGBOOST:
        print("Using XGBoost Classifier engine")
        model = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            random_state=42,
            eval_metric="mlogloss"
        )
    else:
        print("Using GradientBoostingClassifier engine (scikit-learn)")
        model = GradientBoostingClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            random_state=42
        )

    start_train = time.time()
    model.fit(X_train, y_train)
    train_duration = time.time() - start_train

    # Evaluation
    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    report = classification_report(
        y_test, y_pred, target_names=["Tier 1", "Tier 2", "Tier 3"], output_dict=True
    )

    print(f"\nTraining completed in {train_duration:.2f}s")
    print(f"Test Accuracy: {accuracy * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=["Tier 1", "Tier 2", "Tier 3"]))

    # Artifact export
    model_file = export_path / "router_model.joblib"
    tfidf_file = export_path / "tfidf_vectorizer.joblib"
    joblib.dump(model, model_file)
    joblib.dump(tfidf, tfidf_file)
    print(f"Model saved to        : {model_file}")
    print(f"Vectorizer saved to   : {tfidf_file}")

    # Save training report
    summary = {
        "timestamp": time.time(),
        "time_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "engine": "XGBoost" if HAS_XGBOOST else "GradientBoosting",
        "total_samples": total_samples,
        "synthetic_samples": len(X_synthetic),
        "benchmark_samples": bench_count,
        "observation_samples": obs_count,
        "feature_count": int(X_train.shape[1]),
        "accuracy": round(accuracy, 4),
        "tier_distribution": tier_counts,
        "duration_seconds": round(train_duration, 2),
        "per_class_f1": {
            "Tier 1": round(report["Tier 1"]["f1-score"], 3),
            "Tier 2": round(report["Tier 2"]["f1-score"], 3),
            "Tier 3": round(report["Tier 3"]["f1-score"], 3),
        }
    }

    report_path = DATA_DIR / "latest_training_report.json"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Report saved to       : {report_path}")
    print("=" * 65)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train Router Complexity Classifier")
    parser.add_argument("--export-dir", type=str, default=None,
                        help="Target directory to save model artifacts")
    parser.add_argument("--no-benchmarks", action="store_true",
                        help="Disable HuggingFace benchmark ingestion")
    parser.add_argument("--no-observations", action="store_true",
                        help="Disable live observations ingestion")
    parser.add_argument("--min-obs", type=int, default=10,
                        help="Minimum live observations before including in training")
    args = parser.parse_args()

    train_and_export(
        export_dir=args.export_dir,
        include_benchmarks=not args.no_benchmarks,
        include_observations=not args.no_observations,
        min_obs=args.min_obs
    )


if __name__ == "__main__":
    main()
