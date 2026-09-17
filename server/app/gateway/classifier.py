import re
from typing import Dict, Any

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
            "agent_weight": agent_weight
        }

    def predict_score(self, prompt: str, agent_name: str = "General") -> float:
        return self.explain(prompt, agent_name)["score"]

    def explain(self, prompt: str, agent_name: str = "General") -> Dict[str, Any]:
        feats = self.extract_features(prompt, agent_name)
        signals = {
            "length_contribution": 0.0,
            "intent_contribution": 0.0,
            "deep_domain_contribution": 0.0,
            "architecture_contribution": 0.0,
            "algorithm_contribution": 0.0,
            "agent_bias_contribution": round(feats["agent_weight"], 3),
            "instruction_payload": {
                "editing_intent": bool(feats["is_editing"]),
                "coding_intent": bool(feats["is_coding"]),
                "architecture_keywords": bool(feats["has_architecture_keywords"]),
                "algorithm_keywords": bool(feats["has_algorithm_keywords"]),
                "quoted_payload": bool(feats["has_quoted_payload"]),
            },
            "matched_signal_flags": [],
        }
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

        # Rule 1: Active grammar / formatting intent overrides technical payload
        # Rule 2: Active coding with advanced algorithms / ML domain
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

        # Rule 3: General open-ended prompts
        signals["length_contribution"] = round((min(feats["word_count"], 400) / 400) * 0.50, 3)
        score = signals["length_contribution"] + feats["agent_weight"]
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
