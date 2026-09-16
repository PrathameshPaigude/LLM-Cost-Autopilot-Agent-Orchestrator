import time
from collections import deque
from typing import Dict, Any, Optional

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
        self.observations = deque(maxlen=160)

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

    def record_route_observation(
        self,
        tier: str,
        provider: str,
        model: str,
        estimated_cost_usd: float,
        confidence: float,
        complexity_score: float,
        quality_confidence: Optional[float] = None,
    ) -> None:
        self.observations.append({
            "timestamp": time.time(),
            "kind": "route",
            "tier": tier,
            "provider": provider,
            "model": model,
            "estimated_cost_usd": round(estimated_cost_usd, 6),
            "confidence": round(confidence, 3),
            "quality_confidence": quality_confidence,
            "complexity_score": round(complexity_score, 3),
        })

    def record_quality_observation(self, confidence: float, estimated_cost_usd: float) -> None:
        self.observations.append({
            "timestamp": time.time(),
            "kind": "review",
            "tier": "Reviewed outcome",
            "provider": "Reviewer",
            "model": "workflow",
            "estimated_cost_usd": round(estimated_cost_usd, 6),
            "confidence": None,
            "quality_confidence": round(confidence, 3),
            "complexity_score": None,
        })

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

    def get_pareto(self) -> Dict[str, Any]:
        points = list(self.observations)
        route_points = [point for point in points if point["kind"] == "route"]
        review_points = [point for point in points if point["kind"] == "review"]
        return {
            "points": points,
            "route_count": len(route_points),
            "review_count": len(review_points),
            "window_size": len(points),
        }

# Global singleton
telemetry = TelemetryTracker()
