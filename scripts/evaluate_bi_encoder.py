"""Evaluate a bi-encoder (zero-shot or a fine-tuned checkpoint) on all English
development queries.

Reports mean Recall@10, Recall@50, MRR, and NDCG@10, saving per-query and
aggregate results to a JSON file. Defaults to the zero-shot base model and
results/bi_encoder_dev.json; pass --model to point at a fine-tuned checkpoint
directory (e.g. results/checkpoints/loqo_46795_mpnet_triplet) and --output to
avoid overwriting the zero-shot results.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bi_encoder import BiEncoder, MODEL_NAME  # noqa: E402
from data_loader import load_corpus, load_qrels, load_queries  # noqa: E402
from evaluation import ndcg_at_k, recall_at_k, reciprocal_rank  # noqa: E402

SPLIT_DIR = Path(__file__).resolve().parent.parent / "data" / "TaskA" / "development" / "en"
DEFAULT_RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "bi_encoder_dev.json"
K_VALUES = (10, 50)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"Model name or local checkpoint path (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_RESULTS_PATH),
        help=f"Where to save results JSON (default: {DEFAULT_RESULTS_PATH})",
    )
    args = parser.parse_args()
    results_path = Path(args.output)

    queries = load_queries(SPLIT_DIR)
    corpus = load_corpus(SPLIT_DIR)
    qrels = load_qrels(SPLIT_DIR)

    print(f"Loading model: {args.model}")
    bi_encoder = BiEncoder(corpus, model_name=args.model)
    max_k = max(K_VALUES)

    per_query = {}
    for query_id, query_text in queries.items():
        relevant_ids = qrels.get(query_id, [])
        retrieved_ids = [doc_id for doc_id, _score in bi_encoder.rank(query_text, top_k=max_k)]
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
        "model": args.model,
        "num_queries": len(queries),
        "mean": mean_metrics,
        "per_query": per_query,
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Evaluated {len(queries)} queries")
    for k in K_VALUES:
        print(f"Mean Recall@{k}: {mean_metrics[f'recall@{k}']:.3f}")
    print(f"MRR: {mean_metrics['mrr']:.3f}")
    print(f"NDCG@10: {mean_metrics['ndcg@10']:.3f}")
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
