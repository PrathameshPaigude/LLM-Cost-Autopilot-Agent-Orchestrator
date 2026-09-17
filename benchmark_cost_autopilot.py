"""Benchmark the gateway's routing and illustrative cost estimates.

Run from the project directory while the FastAPI server is running:
    python benchmark_cost_autopilot.py

The default run forces local Ollama execution so it does not send prompts to
cloud providers. The application's own routing audit supplies token-based
cost estimates; these are illustrative, not billing records.
"""

import json
import statistics
import time
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx


BASE_URL = "http://localhost:8000"
HEALTH_ENDPOINT = f"{BASE_URL}/api/v1/health"
ROUTE_ENDPOINT = f"{BASE_URL}/api/v1/route"
OUTPUT_FILE = Path(__file__).with_name("benchmark_results.json")

SAMPLE_PROMPTS = [
    {"id": 1, "prompt": "Fix the grammar in this sentence: 'He dont know nothing.'", "expected_tier": "Tier 1"},
    {"id": 2, "prompt": "Summarize the key differences between REST and GraphQL APIs.", "expected_tier": "Tier 2"},
    {"id": 3, "prompt": "Write a Python function to reverse a linked list.", "expected_tier": "Tier 2"},
    {"id": 4, "prompt": "Design a distributed rate-limiting system for a multi-tenant SaaS API, including failure modes.", "expected_tier": "Tier 3"},
    {"id": 5, "prompt": "What is the capital of France?", "expected_tier": "Tier 1"},
    {"id": 6, "prompt": "Explain the CAP theorem and its implications for microservice architecture.", "expected_tier": "Tier 3"},
    {"id": 7, "prompt": 'Convert this JSON to a CSV format: {"a":1,"b":2}', "expected_tier": "Tier 1"},
    {"id": 8, "prompt": "Refactor this function to use async/await instead of callbacks.", "expected_tier": "Tier 2"},
    {"id": 9, "prompt": "Correct the grammar in: 'Write a complex distributed system'", "expected_tier": "Tier 1"},
    {"id": 10, "prompt": "Architect a fault-tolerant event-driven pipeline for real-time fraud detection at scale.", "expected_tier": "Tier 3"},
    {"id": 11, "prompt": "Fix the grammar in this sentence: 'He dont know nothing.'", "expected_tier": "Tier 1", "is_duplicate_of": 1},
    {"id": 12, "prompt": "What is the capital of France?", "expected_tier": "Tier 1", "is_duplicate_of": 5},
    {"id": 13, "prompt": 'Convert this JSON to a CSV format: {"a":1,"b":2}', "expected_tier": "Tier 1", "is_duplicate_of": 7},
]


def tier_matches(actual_tier: str | None, expected_tier: str) -> bool:
    return bool(actual_tier and actual_tier.startswith(expected_tier))


def run_benchmark() -> None:
    results = []
    modes = [mode.strip() for mode in os.getenv("BENCHMARK_MODES", "offline,cloud,auto").split(",") if mode.strip()]
    mode_summaries = {}
    with httpx.Client(timeout=360.0) as client:
        try:
            health = client.get(HEALTH_ENDPOINT)
            health.raise_for_status()
            print(f"Backend health check: {health.status_code}")
        except Exception as error:
            print(f"Could not reach backend at {BASE_URL}: {error}")
            return

        for mode in modes:
            client.delete(f"{BASE_URL}/api/v1/debug/cache")
            mode_results = []
            print(f"\nRunning benchmark mode: {mode}")
            for case in SAMPLE_PROMPTS:
                print(f"Running prompt {case['id']}: {case['prompt'][:60]}...")
                started = time.perf_counter()
                try:
                    response = client.post(
                        ROUTE_ENDPOINT,
                        json={"prompt": case["prompt"], "execution_mode": mode, "debug": True},
                    )
                    response.raise_for_status()
                    data = response.json()
                except Exception as error:
                    print(f"  ERROR: {error}")
                    continue

                audit = data.get("routing_audit", {})
                selected = audit.get("selected", {})
                complexity = audit.get("complexity", {})
                classifier_debug = data.get("classifier_debug") or {}
                actual_cost = float(selected.get("estimated_cost_usd", 0.0))
                frontier_cost = float(audit.get("estimated_gpt4_class_cost_usd", 0.0))
                actual_tier = classifier_debug.get("tier") or complexity.get("tier")
                mode_results.append({
                    "mode": mode,
                    "id": case["id"],
                    "prompt": case["prompt"],
                    "expected_tier": case["expected_tier"],
                    "is_duplicate_of": case.get("is_duplicate_of"),
                    "actual_tier": actual_tier,
                    "complexity_score": complexity.get("score"),
                    "provider": data.get("provider"),
                    "model": data.get("model_name"),
                    "latency_seconds": round(time.perf_counter() - started, 3),
                    "cache_hit": bool(data.get("is_cached", False)),
                    "cache_key_used": data.get("cache_key_used"),
                    "classifier_raw_score": classifier_debug.get("score"),
                    "classifier_matched_signals": classifier_debug.get("signals", {}).get("matched_signal_flags", []),
                    "raw_output_preview": data.get("raw_output_preview", ""),
                    "actual_estimated_cost_usd": round(actual_cost, 6),
                    "frontier_equivalent_cost_usd": round(frontier_cost, 6),
                    "estimated_savings_usd": round(max(0.0, frontier_cost - actual_cost), 6),
                    "cost_basis": audit.get("cost_basis"),
                })
            results.extend(mode_results)
            if mode_results:
                mode_actual = sum(item["actual_estimated_cost_usd"] for item in mode_results)
                mode_frontier = sum(item["frontier_equivalent_cost_usd"] for item in mode_results)
                mode_summaries[mode] = {
                    "successful_requests": len(mode_results),
                    "estimated_cost_saved_percent": round(max(0.0, mode_frontier - mode_actual) / mode_frontier * 100, 2) if mode_frontier else 0.0,
                    "average_latency_seconds": round(statistics.mean(item["latency_seconds"] for item in mode_results), 3),
                    "cache_hit_rate_percent": round(sum(item["cache_hit"] for item in mode_results) / len(mode_results) * 100, 2),
                    "expected_tier_match_percent": round(sum(tier_matches(item["actual_tier"], item["expected_tier"]) for item in mode_results) / len(mode_results) * 100, 2),
                }

    if not results:
        print("No successful requests; no benchmark file was written.")
        return

    total_actual = sum(item["actual_estimated_cost_usd"] for item in results)
    total_frontier = sum(item["frontier_equivalent_cost_usd"] for item in results)
    total_savings = max(0.0, total_frontier - total_actual)
    summary = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "modes": mode_summaries,
        "total_requests": len(results),
        "total_actual_estimated_cost_usd": round(total_actual, 6),
        "total_frontier_equivalent_cost_usd": round(total_frontier, 6),
        "total_estimated_savings_usd": round(total_savings, 6),
        "percent_estimated_cost_saved": round(total_savings / total_frontier * 100, 2) if total_frontier else 0.0,
        "average_latency_seconds": round(statistics.mean(item["latency_seconds"] for item in results), 3),
        "cache_hit_rate_percent": round(sum(item["cache_hit"] for item in results) / len(results) * 100, 2),
        "expected_tier_match_percent": round(sum(tier_matches(item["actual_tier"], item["expected_tier"]) for item in results) / len(results) * 100, 2),
        "disclaimer": "Estimated routing costs from application telemetry; not provider billing data.",
    }
    OUTPUT_FILE.write_text(json.dumps({"summary": summary, "per_request": results}, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        print(f"{key}: {value}")
    mismatches = [item for item in results if not tier_matches(item["actual_tier"], item["expected_tier"])]
    if mismatches:
        print(f"\nTIER MISMATCHES ({len(mismatches)}):")
        for item in mismatches:
            print(f"[{item['mode']}] Prompt {item['id']}: expected={item['expected_tier']} actual={item['actual_tier']} score={item['complexity_score']}")
            print(f"  matched_signals={item['classifier_matched_signals']}")
    duplicates = [item for item in results if item.get("is_duplicate_of")]
    if duplicates:
        print("\nCACHE CHECK — duplicate prompt results:")
        for item in duplicates:
            original = next((candidate for candidate in results if candidate["mode"] == item["mode"] and candidate["id"] == item["is_duplicate_of"]), None)
            keys_match = bool(original and original["cache_key_used"] == item["cache_key_used"])
            print(f"[{item['mode']}] Prompt {item['id']} duplicate_of={item['is_duplicate_of']} cache_hit={item['cache_hit']} keys_match={keys_match}")
    print(f"\nFull results saved to {OUTPUT_FILE}")
    print(
        f'\nResume-safe wording: "Estimated a {summary["percent_estimated_cost_saved"]}% '
        f'cost reduction versus GPT-4o-class baseline across {summary["total_requests"]} '
        'offline benchmark prompts."'
    )


if __name__ == "__main__":
    run_benchmark()