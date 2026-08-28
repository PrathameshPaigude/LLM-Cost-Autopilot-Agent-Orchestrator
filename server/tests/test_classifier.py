import unittest
import sys
import os

# Ensure server package is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from server.app.gateway.classifier import classifier
from server.app.gateway.redactor import redactor
from server.app.gateway.cache import cache

class TestGatewayComponents(unittest.TestCase):
    def test_intent_vs_payload_grammar_differentiation(self):
        """Verify that grammar checking a technical sentence is routed to Tier 1, while coding is Tier 3."""
        coding_prompt = "Write a complex code to train xgboost model with cross-validation"
        grammar_prompt = "Correct the grammar in the sentence: Write a complex code to train xgboost model"

        score_code = classifier.predict_score(coding_prompt, agent_name="CodeAgent")
        score_grammar = classifier.predict_score(grammar_prompt, agent_name="WritingAgent")

        # Grammar should be low complexity (Tier 1 < 0.30)
        self.assertLess(score_grammar, 0.30, f"Grammar score too high: {score_grammar}")
        # Coding should be high complexity (Tier 3 > 0.70)
        self.assertGreaterEqual(score_code, 0.70, f"Coding score too low: {score_code}")

    def test_pii_redaction(self):
        """Verify secret API keys and emails are sanitized."""
        secret_prompt = "Here is my key sk-1234567890abcdef1234567890 and email test@example.com"
        sanitized, was_redacted = redactor.sanitize(secret_prompt)
        
        self.assertTrue(was_redacted)
        self.assertNotIn("sk-1234567890abcdef1234567890", sanitized)
        self.assertIn("[REDACTED_API_KEY]", sanitized)
        self.assertNotIn("test@example.com", sanitized)
        self.assertIn("[REDACTED_EMAIL]", sanitized)

    def test_cache_hit_and_ttl(self):
        """Verify cache storage and retrieval."""
        cache.set("test query", "test answer")
        retrieved = cache.get("test query")
        self.assertEqual(retrieved, "test answer")
        self.assertIsNone(cache.get("nonexistent query"))

if __name__ == "__main__":
    unittest.main()
