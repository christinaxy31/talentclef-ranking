"""Evaluate the BM25 baseline on all English development queries.

Reports mean Recall@10, Recall@50, MRR, and NDCG@10, saving per-query and
aggregate results to results/bm25_dev.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bm25 import BM25  # noqa: E402
from data_loader import load_corpus, load_qrels, load_queries  # noqa: E402
from evaluation import ndcg_at_k, recall_at_k, reciprocal_rank  # noqa: E402

SPLIT_DIR = Path(__file__).resolve().parent.parent / "data" / "TaskA" / "development" / "en"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "bm25_dev.json"
K_VALUES = (10, 50)


def main() -> None:
    queries = load_queries(SPLIT_DIR)
    corpus = load_corpus(SPLIT_DIR)
    qrels = load_qrels(SPLIT_DIR)

    bm25 = BM25(corpus)
    max_k = max(K_VALUES)

    per_query = {}
    for query_id, query_text in queries.items():
        relevant_ids = qrels.get(query_id, [])
        retrieved_ids = [doc_id for doc_id, _score in bm25.rank(query_text, top_k=max_k)]
        per_query[query_id] = {
            f"recall@{k}": recall_at_k(retrieved_ids, relevant_ids, k=k) for k in K_VALUES
        }
        per_query[query_id]["reciprocal_rank"] = reciprocal_rank(retrieved_ids, relevant_ids)
        per_query[query_id]["ndcg@10"] = ndcg_at_k(retrieved_ids, relevant_ids, k=10)

    mean_metrics = {
        f"recall@{k}": sum(r[f"recall@{k}"] for r in per_query.values()) / len(per_query)
        for k in K_VALUES
    }
    mean_metrics["mrr"] = sum(r["reciprocal_rank"] for r in per_query.values()) / len(per_query)
    mean_metrics["ndcg@10"] = sum(r["ndcg@10"] for r in per_query.values()) / len(per_query)

    results = {
        "num_queries": len(queries),
        "mean": mean_metrics,
        "per_query": per_query,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Evaluated {len(queries)} queries")
    for k in K_VALUES:
        print(f"Mean Recall@{k}: {mean_metrics[f'recall@{k}']:.3f}")
    print(f"MRR: {mean_metrics['mrr']:.3f}")
    print(f"NDCG@10: {mean_metrics['ndcg@10']:.3f}")
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
