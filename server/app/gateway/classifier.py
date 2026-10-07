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
            r"\b(correct|grammar|proofread|spelling|rephrase|paraphrase|fix typo|capitalize|format json|clean text)\b",
            re.IGNORECASE
        )
        self.coding_intent = re.compile(
            r"\b(write code|implement|build|develop|create function|create class|pipeline|script|sql query|fastapi|backend|frontend)\b",
            re.IGNORECASE
        )
        self.architecture_keywords = re.compile(
            r"\b(architecture|architect|distributed|microservices?|fault[- ]tolerant|scalable|cap theorem|"
            r"event[- ]driven|rate[- ]limiting|consistency|availability|partition tolerance|"
            r"high[- ]availability|load balancing|rest|graphql|microservice architecture|fraud detection)\b",
            re.IGNORECASE
        )
        self.strong_architecture_keywords = re.compile(
            r"\b(architecture|architect|distributed|microservices?|fault[- ]tolerant|scalable|cap theorem|"
            r"event[- ]driven|rate[- ]limiting|partition tolerance|high[- ]availability|load balancing|at scale)\b",
            re.IGNORECASE
        )
        self.algorithm_keywords = re.compile(
            r"\b(linked list|recursion|sorting|traversal|binary tree|graph|dynamic programming|"
            r"time complexity|space complexity|big[- ]o|async/?await|refactor)\b",
            re.IGNORECASE
        )
        self.high_complexity_domains = re.compile(
            r"\b(xgboost|neural|deep learning|transformer|distributed|concurrency|asyncio|race condition|jwt auth|cryptography|algorithm optimization|sharding|schema migration)\b",
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
        
        # Check first 80 characters for the governing instruction verb
        prefix = prompt_lower[:80]
        is_editing = 1.0 if self.editing_intent.search(prefix) else 0.0
        is_coding = 1.0 if (self.coding_intent.search(prefix) and not is_editing) else 0.0
        has_architecture_keywords = 1.0 if self.architecture_keywords.search(prompt_lower) else 0.0
        has_strong_architecture_keywords = 1.0 if self.strong_architecture_keywords.search(prompt_lower) else 0.0
        has_algorithm_keywords = 1.0 if self.algorithm_keywords.search(prompt_lower) else 0.0
        has_deep_domain = 1.0 if self.high_complexity_domains.search(prompt_lower) else 0.0
        
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

        # --- Short-Prompt Guardrail ---
        # If the prompt is under 25 words and has no complex domain signals,
        # force Tier 1 routing regardless of what the ML model thinks.
        # This prevents trivial questions from hitting slow frontier providers.
        word_count = feats["word_count"]
        has_complexity = (
            feats["has_architecture_keywords"]
            or feats["has_strong_architecture_keywords"]
            or feats["has_algorithm_keywords"]
            or feats["has_deep_domain"]
            or feats["is_coding"]
        )
        force_tier1 = word_count < 25 and not has_complexity

        signals = {
            "length_contribution": 0.0,
            "intent_contribution": 0.0,
            "deep_domain_contribution": 0.0,
            "architecture_contribution": 0.0,
            "algorithm_contribution": 0.0,
            "agent_bias_contribution": round(feats["agent_weight"], 3),
            "classifier_engine": "XGBoost/GBDT ML Model" if self.is_ml_active() else "Deterministic Intent Heuristics",
            "instruction_payload": {
                "editing_intent": bool(feats["is_editing"]),
                "coding_intent": bool(feats["is_coding"]),
                "architecture_keywords": bool(feats["has_architecture_keywords"]),
                "algorithm_keywords": bool(feats["has_algorithm_keywords"]),
                "quoted_payload": bool(feats["has_quoted_payload"]),
            },
            "matched_signal_flags": [],
        }

        # If trained ML model is available and prompt is not an explicit overriding proofread intent, use ML model prediction
        if self.is_ml_active() and not feats["is_editing"]:
            try:
                import numpy as np
                text_vec = self.tfidf_vectorizer.transform([prompt]).toarray()
                # Construct linguistic features matching the trained model (6 features: editing, coding, frontier, quotes, word_ratio, len_ratio)
                ling_vec = np.array([[
                    feats["is_editing"],
                    feats["is_coding"],
                    feats["has_deep_domain"],
                    feats["has_quoted_payload"],
                    min(feats["word_count"] / 150.0, 1.0),
                    min(len(prompt) / 800.0, 1.0),
                ]], dtype=np.float32)
                
                # Check feature alignment with trained model
                expected_n = getattr(self.ml_model, "n_features_in_", 806)
                combined = np.hstack([text_vec, ling_vec])
                if combined.shape[1] != expected_n:
                    # Pad or truncate if feature count differs
                    if combined.shape[1] < expected_n:
                        pad = np.zeros((1, expected_n - combined.shape[1]), dtype=np.float32)
                        combined = np.hstack([combined, pad])
                    else:
                        combined = combined[:, :expected_n]

                probs = self.ml_model.predict_proba(combined)[0]
                # Weighted continuous score: P(Tier 2)*0.50 + P(Tier 3)*1.00 + agent bias
                ml_score = float(probs[1] * 0.50 + probs[2] * 0.95 + feats["agent_weight"] * 0.15)
                ml_score = min(max(ml_score, 0.05), 1.0)
                # Apply short-prompt guardrail clamp
                if force_tier1:
                    ml_score = min(ml_score, 0.20)
                    signals["matched_signal_flags"].append("short_prompt_guardrail")
                if feats["has_deep_domain"]:
                    signals["matched_signal_flags"].append("deep_domain_keyword")
                if feats["is_coding"]:
                    signals["matched_signal_flags"].append("coding_intent")
                signals["ml_class_probabilities"] = {
                    "Tier 1 (Light)": round(float(probs[0]), 3),
                    "Tier 2 (Balanced)": round(float(probs[1]), 3),
                    "Tier 3 (Frontier)": round(float(probs[2]), 3),
                }
                return self._explanation(ml_score, feats, signals)
            except Exception as e:
                logger.debug(f"ML runtime prediction error ({e}), continuing with heuristic tree.")

        # Heuristic Tree Execution
        if feats["is_editing"]:
            signals["matched_signal_flags"].append("editing_intent")
            signals["intent_contribution"] = 0.12
            signals["length_contribution"] = round((min(feats["word_count"], 300) / 300) * 0.10, 3)
            score = min(signals["intent_contribution"] + signals["length_contribution"], 0.28)
            return self._explanation(score, feats, signals)

        if feats["has_strong_architecture_keywords"]:
            signals["matched_signal_flags"].append("architecture_keywords")
            if feats["has_deep_domain"]:
                signals["matched_signal_flags"].append("deep_domain_keyword")
            signals["architecture_contribution"] = 0.72
            signals["length_contribution"] = round((min(feats["word_count"], 200) / 200) * 0.10, 3)
            signals["agent_bias_contribution"] = round(feats["agent_weight"] * 0.2, 3)
            if feats["is_coding"]:
                signals["matched_signal_flags"].append("coding_intent")
                signals["intent_contribution"] = 0.10
            score = signals["architecture_contribution"] + signals["length_contribution"] + signals["agent_bias_contribution"] + signals["intent_contribution"]
            return self._explanation(min(score, 1.00), feats, signals)

        if feats["has_algorithm_keywords"]:
            signals["matched_signal_flags"].append("algorithm_keywords")
            signals["algorithm_contribution"] = 0.42
            signals["length_contribution"] = round((min(feats["word_count"], 200) / 200) * 0.16, 3)
            signals["agent_bias_contribution"] = round(feats["agent_weight"] * 0.2, 3)
            if feats["is_coding"]:
                signals["matched_signal_flags"].append("coding_intent")
                signals["intent_contribution"] = 0.10
            score = signals["algorithm_contribution"] + signals["length_contribution"] + signals["agent_bias_contribution"] + signals["intent_contribution"]
            return self._explanation(min(score, 0.68), feats, signals)

        if feats["has_architecture_keywords"]:
            signals["matched_signal_flags"].append("architecture_keywords")
            signals["architecture_contribution"] = 0.45
            signals["length_contribution"] = round((min(feats["word_count"], 300) / 300) * 0.10, 3)
            score = signals["architecture_contribution"] + signals["length_contribution"] + feats["agent_weight"]
            return self._explanation(min(score, 0.68), feats, signals)

        if feats["is_coding"] == 1.0 or feats["has_deep_domain"] == 1.0:
            if feats["has_deep_domain"] == 1.0:
                signals["matched_signal_flags"].append("deep_domain_keyword")
                signals["deep_domain_contribution"] = 0.75
                signals["length_contribution"] = round((min(feats["word_count"], 200) / 200) * 0.15, 3)
                signals["agent_bias_contribution"] = round(feats["agent_weight"] * 0.2, 3)
                score = signals["deep_domain_contribution"] + signals["length_contribution"] + signals["agent_bias_contribution"]
                return self._explanation(min(score, 1.00), feats, signals)
            else:
                signals["matched_signal_flags"].append("coding_intent")
                signals["intent_contribution"] = 0.40
                signals["length_contribution"] = round((min(feats["word_count"], 200) / 200) * 0.20, 3)
                signals["agent_bias_contribution"] = round(feats["agent_weight"] * 0.2, 3)
                score = signals["intent_contribution"] + signals["length_contribution"] + signals["agent_bias_contribution"]
                return self._explanation(min(score, 0.68), feats, signals)

        signals["length_contribution"] = round((min(feats["word_count"], 400) / 400) * 0.50, 3)
        score = signals["length_contribution"] + feats["agent_weight"]
        # Apply short-prompt guardrail clamp
        if force_tier1:
            score = min(score, 0.20)
            signals["matched_signal_flags"].append("short_prompt_guardrail")
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
