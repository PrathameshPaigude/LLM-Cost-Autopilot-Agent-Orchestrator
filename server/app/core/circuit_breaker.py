import time
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN

    def call(self, func: Callable, *args, fallback_func: Callable = None, **kwargs) -> Any:
        current_time = time.time()

        # Check if we should attempt recovery
        if self.state == "OPEN":
            if current_time - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF-OPEN"
                logger.info("Circuit Breaker entering HALF-OPEN state. Testing primary target.")
            else:
                logger.warning("Circuit Breaker OPEN. Skipping primary and executing fallback.")
                if fallback_func:
                    return fallback_func(*args, **kwargs)
                raise RuntimeError("Circuit breaker is OPEN and no fallback is available.")

        try:
            result = func(*args, **kwargs)
            if self.state == "HALF-OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("Circuit Breaker successfully recovered to CLOSED state.")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            logger.error(f"Primary invocation failed ({self.failure_count}/{self.failure_threshold}): {e}")

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.error("Circuit Breaker tripped to OPEN state.")

            if fallback_func:
                logger.info("Executing failover fallback function.")
                return fallback_func(*args, **kwargs)
            raise e
