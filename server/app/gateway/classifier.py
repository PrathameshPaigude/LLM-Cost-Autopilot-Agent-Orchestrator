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
        has_deep_domain = 1.0 if self.high_complexity_domains.search(prompt_lower) else 0.0
        
        # Check for passive payload (quotes, backticks)
        has_quoted_payload = 1.0 if ('"' in prompt or "'" in prompt or "`" in prompt) else 0.0
        
        words = len(prompt.split())
        agent_weight = self.agent_bias.get(agent_name, 0.15)

        return {
            "is_editing": is_editing,
            "is_coding": is_coding,
            "has_deep_domain": has_deep_domain,
            "has_quoted_payload": has_quoted_payload,
            "word_count": words,
            "agent_weight": agent_weight
        }

    def predict_score(self, prompt: str, agent_name: str = "General") -> float:
        feats = self.extract_features(prompt, agent_name)

        # Rule 1: Active grammar / formatting intent overrides technical payload
        if feats["is_editing"] == 1.0:
            # Low complexity (Tier 1)
            score = 0.12 + (min(feats["word_count"], 300) / 300) * 0.10
            return round(min(score, 0.28), 2)

        # Rule 2: Active coding with advanced algorithms / ML domain
        if feats["is_coding"] == 1.0 or feats["has_deep_domain"] == 1.0:
            if feats["has_deep_domain"] == 1.0:
                score = 0.75 + (min(feats["word_count"], 200) / 200) * 0.15 + feats["agent_weight"] * 0.2
                return round(min(score, 1.00), 2)
            else:
                score = 0.40 + (min(feats["word_count"], 200) / 200) * 0.20 + feats["agent_weight"] * 0.2
                return round(min(score, 0.68), 2)

        # Rule 3: General open-ended prompts
        length_score = (min(feats["word_count"], 400) / 400) * 0.50
        base_score = length_score + feats["agent_weight"]
        return round(float(min(max(base_score, 0.15), 0.90)), 2)

classifier = ComplexityClassifier()
