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

    def test_classifier_explanation_contains_signals_and_tier(self):
        explanation = classifier.explain("Design a distributed async pipeline", agent_name="CodeAgent")

        self.assertGreaterEqual(explanation["score"], 0.0)
        self.assertLessEqual(explanation["score"], 1.0)
        self.assertEqual(explanation["tier"], "Tier 3")
        self.assertIn("deep_domain_keyword", explanation["signals"]["matched_signal_flags"])
        self.assertIn("instruction_payload", explanation["signals"])

    def test_cache_debug_reports_normalized_hash_rules(self):
        cache.clear()
        cache.set("  Repeat   Me ", "cached")

        debug = cache.debug_info("repeat me")

        self.assertEqual(debug["size"], 1)
        self.assertTrue(debug["hit"])
        self.assertEqual(debug["key_algorithm"], "SHA-256")
        self.assertEqual(debug["normalization"], "strip, lowercase, collapse whitespace")

    def test_architecture_and_algorithm_signals_raise_expected_tiers(self):
        cases = [
            ("Summarize the key differences between REST and GraphQL APIs.", "Tier 2", "architecture_keywords"),
            ("Write a Python function to reverse a linked list.", "Tier 2", "algorithm_keywords"),
            ("Explain the CAP theorem and its implications for microservice architecture.", "Tier 3", "architecture_keywords"),
            ("Refactor this function to use async/await instead of callbacks.", "Tier 2", "algorithm_keywords"),
            ("Architect a fault-tolerant event-driven pipeline for real-time fraud detection at scale.", "Tier 3", "architecture_keywords"),
        ]

        for prompt, expected_tier, expected_signal in cases:
            explanation = classifier.explain(prompt)
            self.assertEqual(explanation["tier"], expected_tier, prompt)
            self.assertIn(expected_signal, explanation["signals"]["matched_signal_flags"], prompt)

if __name__ == "__main__":
    unittest.main()
