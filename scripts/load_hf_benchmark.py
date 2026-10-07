"""
HuggingFace Routing Benchmark Loader
=====================================
Downloads and processes LLM routing benchmark datasets from HuggingFace Hub:
  - R2-Bench (JiaqiXue/R2-Bench): Model + token-budget -> quality/cost mapping
  - RoutingCompendium (Wikit/RoutingCompendium-perf): Per-query model performance scores

Outputs:
  data/benchmark_prompts.jsonl   -- prompts with tier labels derived from benchmark scores
  data/model_quality_matrix.json -- per-model quality/cost calibration table

Usage:
  python scripts/load_hf_benchmark.py
  python scripts/load_hf_benchmark.py --source r2bench
  python scripts/load_hf_benchmark.py --source routing_compendium
"""

import os
import sys
import json
import time
import logging
import argparse
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# HuggingFace dataset API base
HF_API = "https://huggingface.co/api/datasets"
HF_DATASETS_URL = "https://datasets-server.huggingface.co/rows"

# Dataset identifiers
R2_BENCH_REPO = "JiaqiXue/R2-Bench"
ROUTING_COMPENDIUM_PERF = "Wikit/RoutingCompendium-perf"
ROUTING_COMPENDIUM_COST = "Wikit/RoutingCompendium-cost"


def fetch_hf_rows(dataset_id: str, config: str = "default", split: str = "train",
                   offset: int = 0, length: int = 100) -> Optional[Dict[str, Any]]:
    """Fetches rows from HuggingFace datasets-server API."""
    params = {
        "dataset": dataset_id,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length
    }
    try:
        resp = requests.get(HF_DATASETS_URL, params=params, timeout=(5, 30))
        if resp.status_code == 200:
            return resp.json()
        logger.warning(f"HF API returned {resp.status_code} for {dataset_id}: {resp.text[:200]}")
        return None
    except Exception as e:
        logger.error(f"Failed to fetch from HF datasets-server: {e}")
        return None


def fetch_dataset_info(dataset_id: str) -> Optional[Dict[str, Any]]:
    """Fetches dataset metadata from HuggingFace API."""
    try:
        resp = requests.get(f"{HF_API}/{dataset_id}", timeout=(5, 15))
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception as e:
        logger.debug(f"Dataset info fetch failed: {e}")
        return None


def load_r2_bench(max_rows: int = 2000) -> List[Dict[str, Any]]:
    """
    Loads R2-Bench dataset and maps entries to our 3-tier routing format.
    R2-Bench contains queries evaluated across 10 LLMs at 16 token budget levels.
    We extract the query text and use the quality/cost relationship to assign tiers.
    """
    logger.info(f"Loading R2-Bench from {R2_BENCH_REPO} (up to {max_rows} rows)...")
    entries = []
    offset = 0
    batch_size = 100

    while offset < max_rows:
        length = min(batch_size, max_rows - offset)
        data = fetch_hf_rows(R2_BENCH_REPO, split="train", offset=offset, length=length)

        if not data or "rows" not in data:
            # Try alternative config names
            for config in ["r2_bench", "R2-Bench", "test"]:
                data = fetch_hf_rows(R2_BENCH_REPO, config=config, split="train",
                                     offset=offset, length=length)
                if data and "rows" in data:
                    break
            if not data or "rows" not in data:
                logger.info(f"No more rows available at offset {offset}, stopping.")
                break

        rows = data["rows"]
        if not rows:
            break

        for row_wrapper in rows:
            row = row_wrapper.get("row", row_wrapper)
            entry = _parse_r2_bench_row(row)
            if entry:
                entries.append(entry)

        offset += len(rows)
        logger.info(f"  Fetched {len(entries)} entries so far...")
        time.sleep(0.3)  # Rate limit courtesy

    logger.info(f"R2-Bench: loaded {len(entries)} benchmark entries.")
    return entries


def _parse_r2_bench_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parses a single R2-Bench row into our training format."""
    query = (
        row.get("original_prompt")
        or row.get("templated_prompt")
        or row.get("query")
        or row.get("prompt")
        or row.get("instruction")
        or row.get("input", "")
    )
    if not query or len(str(query).strip()) < 10:
        return None

    query = str(query).strip()

    # R2-Bench features correctness_score (0.0 - 1.0) and actual_token_count
    correctness_raw = row.get("correctness_score")
    token_count_raw = row.get("actual_token_count")

    correctness = None
    if correctness_raw is not None:
        try:
            correctness = float(correctness_raw)
        except (ValueError, TypeError):
            pass

    token_count = 0
    if token_count_raw is not None:
        try:
            token_count = int(token_count_raw)
        except (ValueError, TypeError):
            pass

    if correctness is not None:
        # High correctness with low/medium tokens = Tier 1 or Tier 2
        # Low correctness or complex query = Tier 3
        if correctness >= 0.85 and token_count <= 250:
            tier = 0  # Tier 1 (Solvable easily by light models)
        elif correctness < 0.40 or token_count > 600:
            tier = 2  # Tier 3 (Hard task requiring deep reasoning)
        else:
            tier = 1  # Tier 2 (Balanced task)
    else:
        tier = _estimate_tier_from_text(query)

    return {
        "prompt": query,
        "tier": tier,
        "source": "r2_bench",
        "correctness_score": correctness,
        "token_count": token_count,
        "key": row.get("key", "")
    }


def _estimate_tier_from_text(text: str) -> int:
    """Fallback tier estimation from text when benchmark scores are missing."""
    import re
    text_lower = text.lower()

    # Tier 1 signals: simple editing, grammar, translation, formatting
    tier1_patterns = re.compile(
        r"\b(correct|grammar|proofread|spelling|translate|format|summarize briefly|"
        r"define|capitalize|convert|remove duplicate)\b", re.IGNORECASE
    )
    # Tier 3 signals: architecture, proofs, distributed systems, deep ML
    tier3_patterns = re.compile(
        r"\b(distributed|fault.tolerant|architecture|consensus|proof|theorem|"
        r"reinforcement learning|transformer|attention mechanism|sharding|"
        r"lock.free|concurrent|zero.knowledge|raft|paxos|eBPF)\b", re.IGNORECASE
    )

    if tier1_patterns.search(text_lower[:100]):
        return 0
    if tier3_patterns.search(text_lower):
        return 2
    return 1


def load_routing_compendium(max_rows: int = 1000) -> List[Dict[str, Any]]:
    """
    Loads RoutingCompendium performance dataset.
    Contains per-query, per-model performance and cost metrics.
    """
    logger.info(f"Loading RoutingCompendium from {ROUTING_COMPENDIUM_PERF} (up to {max_rows} rows)...")
    entries = []
    offset = 0
    batch_size = 100

    while offset < max_rows:
        length = min(batch_size, max_rows - offset)
        data = fetch_hf_rows(ROUTING_COMPENDIUM_PERF, split="train", offset=offset, length=length)

        if not data or "rows" not in data:
            for config in ["default", "perf", "performance"]:
                data = fetch_hf_rows(ROUTING_COMPENDIUM_PERF, config=config, split="train",
                                     offset=offset, length=length)
                if data and "rows" in data:
                    break
            if not data or "rows" not in data:
                break

        rows = data["rows"]
        if not rows:
            break

        for row_wrapper in rows:
            row = row_wrapper.get("row", row_wrapper)
            entry = _parse_compendium_row(row)
            if entry:
                entries.append(entry)

        offset += len(rows)
        time.sleep(0.3)

    logger.info(f"RoutingCompendium: loaded {len(entries)} benchmark entries.")
    return entries


def _parse_compendium_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parses a RoutingCompendium row."""
    query = row.get("query") or row.get("prompt") or row.get("instruction") or row.get("input", "")
    if not query or len(str(query).strip()) < 10:
        return None

    query = str(query).strip()
    tier = _estimate_tier_from_text(query)

    return {
        "prompt": query,
        "tier": tier,
        "source": "routing_compendium",
        "raw_scores": {k: v for k, v in row.items() if isinstance(v, (int, float))}
    }


def build_model_quality_matrix(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates benchmark entries into a per-model quality/cost matrix.
    This is loaded at runtime by the router to calibrate tier assignments.
    """
    model_stats: Dict[str, Dict[str, Any]] = {}

    for entry in entries:
        raw = entry.get("raw_scores", {})
        for key, score in raw.items():
            if not isinstance(score, (int, float)):
                continue
            # Normalize model name keys
            model_key = key.strip().lower().replace(" ", "_")
            if model_key in ("id", "idx", "index", "tier"):
                continue

            if model_key not in model_stats:
                model_stats[model_key] = {
                    "total_score": 0.0,
                    "count": 0,
                    "tier_counts": {0: 0, 1: 0, 2: 0}
                }
            model_stats[model_key]["total_score"] += float(score)
            model_stats[model_key]["count"] += 1
            model_stats[model_key]["tier_counts"][entry["tier"]] += 1

    matrix = {}
    for model_key, stats in model_stats.items():
        if stats["count"] > 0:
            matrix[model_key] = {
                "average_quality": round(stats["total_score"] / stats["count"], 4),
                "sample_count": stats["count"],
                "tier_distribution": stats["tier_counts"]
            }

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_benchmark_entries": len(entries),
        "models": matrix
    }


def export_benchmark_data(entries: List[Dict[str, Any]], matrix: Dict[str, Any]):
    """Writes benchmark data to disk for training pipeline consumption."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Export prompts as JSONL for training pipeline
    prompts_path = DATA_DIR / "benchmark_prompts.jsonl"
    with open(prompts_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info(f"Exported {len(entries)} benchmark prompts to {prompts_path}")

    # Export quality matrix
    matrix_path = DATA_DIR / "model_quality_matrix.json"
    with open(matrix_path, "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2)
    logger.info(f"Exported model quality matrix to {matrix_path}")

    # Print tier distribution
    tier_counts = {0: 0, 1: 0, 2: 0}
    for entry in entries:
        tier_counts[entry["tier"]] += 1
    logger.info(f"Tier distribution: T1={tier_counts[0]}, T2={tier_counts[1]}, T3={tier_counts[2]}")


def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace routing benchmark datasets")
    parser.add_argument("--source", choices=["r2bench", "routing_compendium", "all"],
                        default="all", help="Which benchmark to download")
    parser.add_argument("--max-rows", type=int, default=2000,
                        help="Maximum rows to fetch per dataset")
    args = parser.parse_args()

    all_entries = []

    if args.source in ("r2bench", "all"):
        r2_entries = load_r2_bench(max_rows=args.max_rows)
        all_entries.extend(r2_entries)

    if args.source in ("routing_compendium", "all"):
        rc_entries = load_routing_compendium(max_rows=args.max_rows)
        all_entries.extend(rc_entries)

    if not all_entries:
        logger.warning(
            "No benchmark data was retrieved from HuggingFace. "
            "This may be due to network issues or dataset access restrictions. "
            "The router will continue using synthetic training data and live observations."
        )
        # Create empty files so the training pipeline doesn't error
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "benchmark_prompts.jsonl").write_text("")
        (DATA_DIR / "model_quality_matrix.json").write_text(json.dumps({
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_benchmark_entries": 0,
            "models": {},
            "note": "No benchmark data available. Using synthetic training data only."
        }))
        return

    matrix = build_model_quality_matrix(all_entries)
    export_benchmark_data(all_entries, matrix)

    logger.info(f"Done. Total entries: {len(all_entries)}")
    logger.info(f"Model quality matrix covers {len(matrix.get('models', {}))} model identifiers.")


if __name__ == "__main__":
    main()
