import os
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ComplexityClassifier:
    def __init__(self):
        # Priority Intent Regexes
        self.editing_intent = re.compile(
            r"\b("
            r"correct|grammar|proofread|spelling|rephrase|paraphrase|fix typos?|capitalize|clean text|"
            r"make concise|uppercase|lowercase|summarize in \d+ (bullet|sentence|word)s?|translate to \w+|"
            r"format (this |the )?(json|text|markdown|code|payload|data)?|beautify|prettify|indent"
            r")\b",
            re.IGNORECASE
        )
        self.simple_qa_intent = re.compile(
            r"^(hi|hello|hey|who is|who was|what is the capital of|define \w+|what does \w+ stand for)\b",
            re.IGNORECASE
        )
        self.coding_intent = re.compile(
            r"\b("
            r"write (a |an |the |some )?(python|javascript|typescript|golang|go|rust|java|c\+\+|c#|sql|bash|shell|html|css)?\s*(code|script|program|function|class|method|module|component|api|endpoint|query|scraper|regex|parser|algorithm)|"
            r"implement|develop|create function|create class|build an? api|fastapi|backend|frontend|react|vue|angular|django|flask|express|spring boot|"
            r"unit test|pytest|dockerfile|docker-compose|sql query|database schema|crud|"
            r"debug|fix (the |this )?(bug|error|exception|issue|syntax error)|"
            r"refactor|optimize (the |this )?(code|function|query|algorithm)"
            r")\b",
            re.IGNORECASE
        )
        self.strong_architecture_keywords = re.compile(
            r"\b("
            r"distributed systems?|microservices? architecture|fault[- ]tolerant|scalable architecture|"
            r"cap theorem|event[- ]driven architecture|multi[- ]region|active[- ]active|raft consensus|"
            r"paxos|byzantine fault|consensus protocol|sharding|consistent hashing|data partition(ing)?|"
            r"high[- ]availability|load balancing|disaster recovery|100k tps|at scale|"
            r"zero[- ]knowledge|zk[- ]snarks?|formal verification|mathematical proof|"
            r"derive (from first principles|the mathematical proof)|"
            r"ebpf tracing|kernel module|cuda kernel|actor[- ]critic|transformer architecture"
            r")\b",
            re.IGNORECASE
        )
        self.architecture_keywords = re.compile(
            r"\b("
            r"architecture|architect|distributed|microservices?|fault[- ]tolerant|scalable|cap theorem|"
            r"event[- ]driven|rate[- ]limiting|consistency|availability|partition tolerance|"
            r"high[- ]availability|load balancing|rest|graphql|database replication|message broker|"
            r"kafka|rabbitmq|redis cluster|system design|reverse proxy|gateway architecture"
            r")\b",
            re.IGNORECASE
        )
        self.algorithm_keywords = re.compile(
            r"\b("
            r"binary search( tree)?|bst|linked list|doubly linked list|sorting|quicksort|mergesort|heapsort|"
            r"dijkstra|bellman-ford|a\* search|bfs|dfs|breadth-first|depth-first|tree traversal|"
            r"graph algorithm|dynamic programming|memoization|recursion|backtracking|"
            r"two pointers?|sliding window|bit manipulation|binary tree|avl tree|red-black tree|"
            r"trie|hash ?table|hash ?map|priority queue|heapq|stack|queue|deque|"
            r"time complexity|space complexity|big[- ]o|asymptotic|async/?await"
            r")\b",
            re.IGNORECASE
        )
        self.high_complexity_domains = re.compile(
            r"\b("
            r"xgboost|neural network|deep learning|transformer|distributed consensus|concurrency|"
            r"asyncio|race condition|jwt auth|cryptography|algorithm optimization|sharding|schema migration|"
            r"zero[- ]knowledge|zk[- ]snark|raft|paxos|byzantine|formal proof"
            r")\b",
            re.IGNORECASE
        )

        # Agent persona bias weights
        self.agent_bias = {
            "Supervisor": 0.05,
            "WritingAgent": 0.10,
            "AnalysisAgent": 0.35,
            "ResearchAgent": 0.25,
            "CodeAgent": 0.40,
            "CVSpecialist": 0.30,
            "Reviewer": 0.35
        }

        # Optional ML model artifact loading (XGBoost / GradientBoosting)
        self.ml_model = None
        self.tfidf_vectorizer = None
        self._load_trained_ml_model()

        # Benchmark calibration table (derived from HF R2-Bench / RoutingCompendium)
        self.benchmark_matrix: Dict[str, Any] = {}
        self._load_benchmark_matrix()

    def _load_trained_ml_model(self):
        try:
            import joblib
            gateway_dir = Path(__file__).parent
            model_path = gateway_dir / "router_model.joblib"
            tfidf_path = gateway_dir / "tfidf_vectorizer.joblib"

            if model_path.exists() and tfidf_path.exists():
                self.ml_model = joblib.load(str(model_path))
                self.tfidf_vectorizer = joblib.load(str(tfidf_path))
                logger.info("Successfully loaded pre-trained ML Router Model (XGBoost/GBDT) into ComplexityClassifier.")
        except Exception as e:
            logger.debug(f"ML model loading bypassed ({e}); using deterministic heuristic classifier engine.")

    def _load_benchmark_matrix(self):
        try:
            matrix_path = Path(__file__).resolve().parents[3] / "data" / "model_quality_matrix.json"
            if matrix_path.exists():
                import json
                with open(matrix_path, "r", encoding="utf-8") as f:
                    self.benchmark_matrix = json.load(f)
                logger.info(f"Loaded benchmark quality matrix with {len(self.benchmark_matrix.get('models', {}))} model calibrations.")
        except Exception as e:
            logger.debug(f"Benchmark quality matrix loading bypassed: {e}")

    def is_ml_active(self) -> bool:
        return self.ml_model is not None and self.tfidf_vectorizer is not None

    def get_benchmark_status(self) -> Dict[str, Any]:
        data_dir = Path(__file__).resolve().parents[3] / "data"
        benchmark_file = data_dir / "benchmark_prompts.jsonl"
        
        benchmark_rows = 0
        if benchmark_file.exists():
            try:
                with open(benchmark_file, "r", encoding="utf-8") as f:
                    benchmark_rows = sum(1 for line in f if line.strip())
            except Exception:
                pass

        return {
            "benchmarks_loaded": bool(self.benchmark_matrix),
            "benchmark_dataset_rows": benchmark_rows,
            "calibrated_models": list(self.benchmark_matrix.get("models", {}).keys()),
            "sources": self.benchmark_matrix.get("sources", ["R2-Bench (JiaqiXue/R2-Bench)", "RoutingCompendium (Wikit)"]),
            "last_calibrated": self.benchmark_matrix.get("generated_at")
        }

    def extract_features(self, prompt: str, agent_name: str = "General") -> Dict[str, Any]:
        prompt_lower = prompt.lower()
        
        is_editing = 1.0 if (self.editing_intent.search(prompt_lower) or self.simple_qa_intent.search(prompt_lower)) else 0.0
        is_coding = 1.0 if (self.coding_intent.search(prompt_lower) and not is_editing) else 0.0
        has_architecture_keywords = 1.0 if (self.architecture_keywords.search(prompt_lower) and not is_editing) else 0.0
        has_strong_architecture_keywords = 1.0 if (self.strong_architecture_keywords.search(prompt_lower) and not is_editing) else 0.0
        has_algorithm_keywords = 1.0 if (self.algorithm_keywords.search(prompt_lower) and not is_editing) else 0.0
        has_deep_domain = 1.0 if (self.high_complexity_domains.search(prompt_lower) and not is_editing) else 0.0
        
        # Check for passive payload (quotes, backticks)
        has_quoted_payload = 1.0 if ('"' in prompt or "'" in prompt or "`" in prompt) else 0.0
        
        words = len(prompt.split())
        agent_weight = self.agent_bias.get(agent_name, 0.15)

        return {
            "is_editing": is_editing,
            "is_coding": is_coding,
            "has_architecture_keywords": has_architecture_keywords,
            "has_strong_architecture_keywords": has_strong_architecture_keywords,
            "has_algorithm_keywords": has_algorithm_keywords,
            "has_deep_domain": has_deep_domain,
            "has_quoted_payload": has_quoted_payload,
            "word_count": words,
            "agent_weight": agent_weight,
            "ml_engine_active": self.is_ml_active()
        }

    def predict_score(self, prompt: str, agent_name: str = "General") -> float:
        return self.explain(prompt, agent_name)["score"]

    def explain(self, prompt: str, agent_name: str = "General") -> Dict[str, Any]:
        feats = self.extract_features(prompt, agent_name)
        word_count = feats["word_count"]

        signals = {
            "length_contribution": 0.0,
            "intent_contribution": 0.0,
            "deep_domain_contribution": 0.0,
            "architecture_contribution": 0.0,
            "algorithm_contribution": 0.0,
            "agent_bias_contribution": round(feats["agent_weight"], 3),
            "classifier_engine": "Hybrid Ensemble (Semantic Rules + ML Model)" if self.is_ml_active() else "Deterministic Intent Heuristics",
            "instruction_payload": {
                "editing_intent": bool(feats["is_editing"]),
                "coding_intent": bool(feats["is_coding"]),
                "architecture_keywords": bool(feats["has_architecture_keywords"]),
                "algorithm_keywords": bool(feats["has_algorithm_keywords"]),
                "quoted_payload": bool(feats["has_quoted_payload"]),
            },
            "matched_signal_flags": [],
        }

        # Optional ML inference for probabilistic telemetry
        ml_probs = None
        if self.is_ml_active():
            try:
                import numpy as np
                text_vec = self.tfidf_vectorizer.transform([prompt]).toarray()
                ling_vec = np.array([[
                    feats["is_editing"],
                    feats["is_coding"],
                    feats["has_deep_domain"],
                    feats["has_quoted_payload"],
                    min(feats["word_count"] / 150.0, 1.0),
                    min(len(prompt) / 800.0, 1.0),
                ]], dtype=np.float32)

                expected_n = getattr(self.ml_model, "n_features_in_", 806)
                combined = np.hstack([text_vec, ling_vec])
                if combined.shape[1] != expected_n:
                    if combined.shape[1] < expected_n:
                        pad = np.zeros((1, expected_n - combined.shape[1]), dtype=np.float32)
                        combined = np.hstack([combined, pad])
                    else:
                        combined = combined[:, :expected_n]

                probs = self.ml_model.predict_proba(combined)[0]
                ml_probs = probs
                signals["ml_class_probabilities"] = {
                    "Tier 1 (Light)": round(float(probs[0]), 3),
                    "Tier 2 (Balanced)": round(float(probs[1]), 3),
                    "Tier 3 (Frontier)": round(float(probs[2]), 3),
                }
            except Exception as e:
                logger.debug(f"ML runtime prediction error ({e})")

        # --- Rule 1: Explicit Editing / Formatting / Simple QA -> Tier 1 Guaranteed (0.10 - 0.28) ---
        if feats["is_editing"]:
            signals["matched_signal_flags"].append("editing_intent")
            signals["intent_contribution"] = 0.12
            signals["length_contribution"] = round((min(feats["word_count"], 300) / 300) * 0.08, 3)
            score = min(signals["intent_contribution"] + signals["length_contribution"], 0.25)
            return self._explanation(score, feats, signals)

        # --- Rule 2: Strong Architecture / Frontier / Deep Math Proof -> Tier 3 Guaranteed (0.75 - 0.98) ---
        if feats["has_strong_architecture_keywords"] or feats["has_deep_domain"]:
            if feats["has_strong_architecture_keywords"]:
                signals["matched_signal_flags"].append("strong_architecture_keywords")
                signals["architecture_contribution"] = 0.76
            if feats["has_deep_domain"]:
                signals["matched_signal_flags"].append("deep_domain_keyword")
                signals["deep_domain_contribution"] = 0.75
            signals["length_contribution"] = round((min(feats["word_count"], 200) / 200) * 0.10, 3)
            signals["agent_bias_contribution"] = round(feats["agent_weight"] * 0.15, 3)
            base_score = max(signals["architecture_contribution"], signals["deep_domain_contribution"])
            score = base_score + signals["length_contribution"] + signals["agent_bias_contribution"]
            return self._explanation(min(max(score, 0.78), 0.98), feats, signals)

        # --- Rule 3: Coding / Algorithms / Implementation -> Tier 2 Guaranteed (0.38 - 0.68) ---
        if feats["is_coding"] or feats["has_algorithm_keywords"]:
            if feats["is_coding"]:
                signals["matched_signal_flags"].append("coding_intent")
                signals["intent_contribution"] = 0.42
            if feats["has_algorithm_keywords"]:
                signals["matched_signal_flags"].append("algorithm_keywords")
                signals["algorithm_contribution"] = 0.45
            signals["length_contribution"] = round((min(feats["word_count"], 200) / 200) * 0.12, 3)
            signals["agent_bias_contribution"] = round(feats["agent_weight"] * 0.15, 3)
            base = max(signals["intent_contribution"], signals["algorithm_contribution"])
            score = base + signals["length_contribution"] + signals["agent_bias_contribution"]
            return self._explanation(min(max(score, 0.42), 0.68), feats, signals)

        # --- Rule 4: Standard Architecture / System Design -> Tier 2 / 3 (0.55 - 0.72) ---
        if feats["has_architecture_keywords"]:
            signals["matched_signal_flags"].append("architecture_keywords")
            signals["architecture_contribution"] = 0.50
            signals["length_contribution"] = round((min(feats["word_count"], 300) / 300) * 0.12, 3)
            score = signals["architecture_contribution"] + signals["length_contribution"] + feats["agent_weight"] * 0.15
            return self._explanation(min(score, 0.72), feats, signals)

        # --- Rule 5: Short-Prompt Guardrail for Trivial General Queries ---
        has_any_signal = (
            feats["has_architecture_keywords"]
            or feats["has_strong_architecture_keywords"]
            or feats["has_algorithm_keywords"]
            or feats["has_deep_domain"]
            or feats["is_coding"]
        )
        if word_count < 15 and not has_any_signal:
            signals["matched_signal_flags"].append("short_prompt_guardrail")
            return self._explanation(0.18, feats, signals)

        # --- Rule 6: Hybrid ML & Length Scaling for Unstructured General Prompts ---
        if ml_probs is not None:
            # Calibrated expectation across tiers
            p0, p1, p2 = float(ml_probs[0]), float(ml_probs[1]), float(ml_probs[2])
            score = p0 * 0.18 + p1 * 0.52 + p2 * 0.85 + (feats["agent_weight"] * 0.08)
        else:
            signals["length_contribution"] = round((min(feats["word_count"], 400) / 400) * 0.50, 3)
            score = 0.20 + signals["length_contribution"] + feats["agent_weight"] * 0.10

        return self._explanation(min(max(score, 0.15), 0.90), feats, signals)

    @staticmethod
    def _explanation(score: float, features: Dict[str, Any], signals: Dict[str, Any]) -> Dict[str, Any]:
        rounded_score = round(float(score), 2)
        tier = "Tier 1" if rounded_score < 0.30 else "Tier 2" if rounded_score <= 0.70 else "Tier 3"
        return {
            "score": rounded_score,
            "tier": tier,
            "features": features,
            "signals": signals,
            "raw_score": rounded_score,
            "matched_signals": signals["matched_signal_flags"],
            "final_tier": tier,
        }

classifier = ComplexityClassifier()
