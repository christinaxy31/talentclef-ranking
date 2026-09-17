"""Compare hard-negative and random-negative fine-tuning (fixed-step LOQO) against baselines.

Reads up to four existing results files and prints mean ± sample-stdev per
metric, all computed over the same 10 English development queries:

- results/bm25_dev.json                    BM25 baseline
- results/bi_encoder_dev.json              zero-shot MPNet bi-encoder
- results/loqo_full_sweep.json             hard-negative fine-tuned (per-fold held-out results)
- results/loqo_full_sweep_random_neg.json  random-negative fine-tuned (per-fold held-out results)

The random-negative file is produced by scripts/train_bi_encoder_loqo_random_neg.py.
If it hasn't been run yet, that comparison is reported as unavailable rather
than approximated from old anecdotal single-fold runs.
"""

import json
import statistics
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
BM25_PATH = RESULTS_DIR / "bm25_dev.json"
ZERO_SHOT_PATH = RESULTS_DIR / "bi_encoder_dev.json"
HARD_NEGATIVE_PATH = RESULTS_DIR / "loqo_full_sweep.json"
RANDOM_NEGATIVE_PATH = RESULTS_DIR / "loqo_full_sweep_random_neg.json"

METRICS = ["recall@10", "recall@50", "mrr", "ndcg@10"]


def per_query_metrics_from_baseline(path):
    """bm25_dev.json / bi_encoder_dev.json store per_query metrics keyed by
    "reciprocal_rank" instead of "mrr" - normalize to a common key set."""
    data = json.loads(path.read_text(encoding="utf-8"))
    normalized = {}
    for query_id, metrics in data["per_query"].items():
        normalized[query_id] = {
            "recall@10": metrics["recall@10"],
            "recall@50": metrics["recall@50"],
            "mrr": metrics["reciprocal_rank"],
            "ndcg@10": metrics["ndcg@10"],
        }
    return normalized


def per_query_metrics_from_loqo_sweep(path):
    """loqo_full_sweep.json is a list of per-fold dicts with a "fine_tuned" field."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {fold["held_out_query_id"]: fold["fine_tuned"] for fold in data}


def mean_std(values):
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def print_method(name, per_query, query_ids):
    missing = [qid for qid in query_ids if qid not in per_query]
    if missing:
        print(f"{name}: SKIPPED - missing results for queries {missing}")
        return
    print(f"\n{name}  (n={len(query_ids)} queries)")
    for metric in METRICS:
        values = [per_query[qid][metric] for qid in query_ids]
        mean, std = mean_std(values)
        print(f"  {metric:10} {mean:.3f} ± {std:.3f}")


def main() -> None:
    if not HARD_NEGATIVE_PATH.exists():
        print(f"{HARD_NEGATIVE_PATH} not found - run scripts/train_bi_encoder_loqo.py first.")
        return

    hard_negative = per_query_metrics_from_loqo_sweep(HARD_NEGATIVE_PATH)
    query_ids = sorted(hard_negative.keys())
    print(f"Comparing over {len(query_ids)} held-out queries: {query_ids}")

    if BM25_PATH.exists():
        print_method("BM25", per_query_metrics_from_baseline(BM25_PATH), query_ids)
    else:
        print(f"\nBM25: SKIPPED - {BM25_PATH} not found")

    if ZERO_SHOT_PATH.exists():
        print_method("Zero-shot MPNet", per_query_metrics_from_baseline(ZERO_SHOT_PATH), query_ids)
    else:
        print(f"\nZero-shot MPNet: SKIPPED - {ZERO_SHOT_PATH} not found")

    print_method("Hard-negative fine-tuned (fixed-step LOQO)", hard_negative, query_ids)

    if RANDOM_NEGATIVE_PATH.exists():
        random_negative = per_query_metrics_from_loqo_sweep(RANDOM_NEGATIVE_PATH)
        print_method("Random-negative fine-tuned (fixed-step LOQO)", random_negative, query_ids)
    else:
        print(
            f"\nRandom-negative fine-tuned: SKIPPED - {RANDOM_NEGATIVE_PATH} not found "
            "(run scripts/train_bi_encoder_loqo_random_neg.py first)"
        )


if __name__ == "__main__":
    main()
