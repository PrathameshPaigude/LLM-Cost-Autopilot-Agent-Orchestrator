import re
from typing import Tuple

class PIIRedactor:
    def __init__(self):
        self.patterns = [
            # API Keys & Secrets (OpenAI, Anthropic, AWS, Generic)
            (re.compile(r"sk-[a-zA-Z0-9_-]{20,}", re.IGNORECASE), "[REDACTED_API_KEY]"),
            (re.compile(r"AKIA[0-9A-Z]{16}", re.IGNORECASE), "[REDACTED_AWS_KEY]"),
            (re.compile(r"ghp_[a-zA-Z0-9]{36}", re.IGNORECASE), "[REDACTED_GITHUB_TOKEN]"),
            (re.compile(r"(?:api_key|apikey|secret|password|bearer|auth)\s*[:=]\s*['\"][^'\"]+['\"]", re.IGNORECASE), "[REDACTED_SECRET]"),
            # Emails
            (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[REDACTED_EMAIL]"),
            # Phone numbers
            (re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[REDACTED_PHONE]"),
            # Credit Card numbers
            (re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"), "[REDACTED_CC]")
        ]

    def sanitize(self, text: str) -> Tuple[str, bool]:
        sanitized_text = text
        was_redacted = False

        for pattern, replacement in self.patterns:
            if pattern.search(sanitized_text):
                sanitized_text = pattern.sub(replacement, sanitized_text)
                was_redacted = True

        return sanitized_text, was_redacted

redactor = PIIRedactor()
