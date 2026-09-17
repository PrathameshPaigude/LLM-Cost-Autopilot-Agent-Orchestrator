import unittest

from server.app.core.config import settings
from server.app.gateway.router import router


class TestRouterModes(unittest.TestCase):
    def test_tier_timeouts_increase_with_complexity(self):
        tier1 = router._timeout_for_score(settings.ROUTING_TIER1_MAX - 0.01)
        tier2 = router._timeout_for_score(settings.ROUTING_TIER1_MAX + 0.01)
        tier3 = router._timeout_for_score(settings.ROUTING_TIER2_MAX + 0.01)

        self.assertLess(tier1, tier2)
        self.assertLess(tier2, tier3)

    def test_invalid_execution_mode_is_rejected_before_provider_call(self):
        with self.assertRaises(ValueError):
            router.route_and_execute("test", execution_mode="not-a-mode")


if __name__ == "__main__":
    unittest.main()