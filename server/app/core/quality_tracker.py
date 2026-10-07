"""
Adaptive Quality Tracker
========================
Passively records every routing decision (model, latency, tokens, cost,
prompt category) to a local JSONL file. This data accumulates automatically
and is consumed by the training pipeline on the next retrain cycle.

Zero user effort required -- every call to `record()` appends one observation.
"""

import json
import time
import threading
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_OBSERVATIONS_FILE = _DATA_DIR / "routing_observations.jsonl"
_FEEDBACK_FILE = _DATA_DIR / "user_feedback.jsonl"


class QualityTracker:
    def __init__(self):
        self._lock = threading.Lock()
        _DATA_DIR.mkdir(parents=True, exist_ok=True)

        # In-memory aggregations for live API queries
        self._model_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_requests": 0,
            "total_latency_ms": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_usd": 0.0,
            "tier_counts": defaultdict(int),
            "feedback_sum": 0.0,
            "feedback_count": 0,
            "errors": 0,
            "last_used": None
        })
        self._total_observations = 0
        self._total_feedback = 0

        # Load existing counts from disk
        self._load_existing_counts()

    def _load_existing_counts(self):
        """Counts existing observations on startup for stats."""
        try:
            if _OBSERVATIONS_FILE.exists():
                with open(_OBSERVATIONS_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            self._total_observations += 1
                            try:
                                obs = json.loads(line)
                                model = obs.get("model", "unknown")
                                self._model_stats[model]["total_requests"] += 1
                                self._model_stats[model]["total_latency_ms"] += obs.get("latency_ms", 0)
                                self._model_stats[model]["total_input_tokens"] += obs.get("input_tokens", 0)
                                self._model_stats[model]["total_output_tokens"] += obs.get("output_tokens", 0)
                                self._model_stats[model]["total_cost_usd"] += obs.get("cost_usd", 0)
                                tier = obs.get("tier", "unknown")
                                self._model_stats[model]["tier_counts"][tier] += 1
                                self._model_stats[model]["last_used"] = obs.get("timestamp")
                            except (json.JSONDecodeError, KeyError):
                                pass
                logger.info(f"QualityTracker: loaded {self._total_observations} existing observations.")
        except Exception as e:
            logger.debug(f"QualityTracker cold start (no prior observations): {e}")

        try:
            if _FEEDBACK_FILE.exists():
                with open(_FEEDBACK_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            self._total_feedback += 1
                            try:
                                fb = json.loads(line)
                                model = fb.get("model", "unknown")
                                rating = fb.get("rating", 0)
                                self._model_stats[model]["feedback_sum"] += rating
                                self._model_stats[model]["feedback_count"] += 1
                            except (json.JSONDecodeError, KeyError):
                                pass
        except Exception:
            pass

    def record(
        self,
        provider: str,
        model: str,
        tier: str,
        complexity_score: float,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        prompt_category: Optional[str] = None,
        success: bool = True,
        cache_hit: bool = False
    ):
        """Records a single routing observation. Called after every gateway dispatch."""
        observation = {
            "timestamp": time.time(),
            "time_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "provider": provider,
            "model": model,
            "tier": tier,
            "complexity_score": round(complexity_score, 4),
            "latency_ms": round(latency_ms, 1),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost_usd, 8),
            "prompt_category": prompt_category,
            "success": success,
            "cache_hit": cache_hit
        }

        with self._lock:
            try:
                with open(_OBSERVATIONS_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(observation, ensure_ascii=False) + "\n")
                self._total_observations += 1

                stats = self._model_stats[model]
                stats["total_requests"] += 1
                stats["total_latency_ms"] += latency_ms
                stats["total_input_tokens"] += input_tokens
                stats["total_output_tokens"] += output_tokens
                stats["total_cost_usd"] += cost_usd
                stats["tier_counts"][tier] += 1
                stats["last_used"] = observation["time_iso"]
                if not success:
                    stats["errors"] += 1
            except Exception as e:
                logger.debug(f"QualityTracker record failed: {e}")

    def record_feedback(
        self,
        model: str,
        provider: str,
        rating: int,
        prompt_snippet: Optional[str] = None,
        comment: Optional[str] = None
    ):
        """Records user quality feedback (1-5 scale) for a specific response."""
        feedback = {
            "timestamp": time.time(),
            "time_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model": model,
            "provider": provider,
            "rating": max(1, min(5, rating)),
            "prompt_snippet": (prompt_snippet or "")[:200],
            "comment": (comment or "")[:500]
        }

        with self._lock:
            try:
                with open(_FEEDBACK_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(feedback, ensure_ascii=False) + "\n")
                self._total_feedback += 1

                stats = self._model_stats[model]
                stats["feedback_sum"] += feedback["rating"]
                stats["feedback_count"] += 1
            except Exception as e:
                logger.debug(f"QualityTracker feedback write failed: {e}")

    def get_performance_matrix(self) -> Dict[str, Any]:
        """Returns aggregated per-model performance stats for the API."""
        models = {}
        for model, stats in self._model_stats.items():
            if stats["total_requests"] == 0:
                continue
            avg_latency = stats["total_latency_ms"] / stats["total_requests"]
            avg_feedback = (
                round(stats["feedback_sum"] / stats["feedback_count"], 2)
                if stats["feedback_count"] > 0 else None
            )
            models[model] = {
                "total_requests": stats["total_requests"],
                "avg_latency_ms": round(avg_latency, 1),
                "total_input_tokens": stats["total_input_tokens"],
                "total_output_tokens": stats["total_output_tokens"],
                "total_cost_usd": round(stats["total_cost_usd"], 6),
                "avg_cost_per_request_usd": round(stats["total_cost_usd"] / stats["total_requests"], 8),
                "error_rate": round(stats["errors"] / stats["total_requests"], 4),
                "tier_distribution": dict(stats["tier_counts"]),
                "avg_user_rating": avg_feedback,
                "feedback_count": stats["feedback_count"],
                "last_used": stats["last_used"]
            }

        return {
            "total_observations": self._total_observations,
            "total_feedback_entries": self._total_feedback,
            "training_ready": self._total_observations >= 50,
            "recommended_retrain_at": 200,
            "models": models
        }

    def get_training_observations(self) -> List[Dict[str, Any]]:
        """Returns all raw observations for the training pipeline."""
        observations = []
        try:
            if _OBSERVATIONS_FILE.exists():
                with open(_OBSERVATIONS_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                observations.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
        except Exception as e:
            logger.error(f"Failed to read observations: {e}")
        return observations

    @property
    def observation_count(self) -> int:
        return self._total_observations

    @property
    def feedback_count(self) -> int:
        return self._total_feedback


# Global singleton
quality_tracker = QualityTracker()
