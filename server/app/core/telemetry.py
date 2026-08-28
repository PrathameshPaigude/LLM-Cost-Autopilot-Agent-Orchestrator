import time
from typing import Dict, Any

class TelemetryTracker:
    def __init__(self):
        self.total_requests: int = 0
        self.cache_hits: int = 0
        self.tier1_calls: int = 0
        self.tier2_calls: int = 0
        self.tier3_calls: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.simulated_frontier_cost: float = 0.0
        self.actual_cost: float = 0.0

    def record_request(self, tier: str, input_text: str, output_text: str, is_cache: bool = False):
        self.total_requests += 1
        
        # Estimate token count (1 token ≈ 4 characters or 0.75 words)
        in_tokens = max(1, len(input_text.split()) * 4 // 3)
        out_tokens = max(1, len(output_text.split()) * 4 // 3)
        
        self.total_input_tokens += in_tokens
        self.total_output_tokens += out_tokens

        # Claude 3.5 Sonnet / Opus baseline ($15/1M in, $75/1M out)
        baseline_cost = (in_tokens * 15.0 / 1_000_000) + (out_tokens * 75.0 / 1_000_000)
        self.simulated_frontier_cost += baseline_cost

        if is_cache:
            self.cache_hits += 1
        elif "Tier 1" in tier:
            self.tier1_calls += 1
        elif "Tier 2" in tier:
            self.tier2_calls += 1
        elif "Tier 3" in tier:
            self.tier3_calls += 1

    def get_summary(self) -> Dict[str, Any]:
        dollars_saved = max(0.0, self.simulated_frontier_cost - self.actual_cost)
        # Compute conservation estimate: ~0.0015 Wh per Tier 1 query vs ~0.08 Wh per 70B/Frontier query
        watt_hours_saved = (self.tier1_calls * 0.078) + (self.cache_hits * 0.080)
        
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "tier1_calls": self.tier1_calls,
            "tier2_calls": self.tier2_calls,
            "tier3_calls": self.tier3_calls,
            "total_tokens_processed": self.total_input_tokens + self.total_output_tokens,
            "simulated_frontier_cost_usd": round(self.simulated_frontier_cost, 4),
            "actual_cost_usd": round(self.actual_cost, 4),
            "net_dollars_saved_usd": round(dollars_saved, 4),
            "energy_saved_watt_hours": round(watt_hours_saved, 2)
        }

# Global singleton
telemetry = TelemetryTracker()
