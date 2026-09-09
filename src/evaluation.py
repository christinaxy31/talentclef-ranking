"""Minimal retrieval evaluation: Recall@K, MRR, and NDCG@K."""

import math
from typing import Iterable


def recall_at_k(retrieved_ids: Iterable[str], relevant_ids: Iterable[str], k: int = 10) -> float:
    """Fraction of relevant_ids found within the first k retrieved_ids."""
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0
    top_k = list(retrieved_ids)[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / len(relevant_set)


def reciprocal_rank(retrieved_ids: Iterable[str], relevant_ids: Iterable[str]) -> float:
    """1 / rank of the first relevant id in retrieved_ids (0.0 if none found)."""
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Iterable[str], relevant_ids: Iterable[str], k: int = 10) -> float:
    """Normalized DCG@k for binary relevance: DCG@k / IDCG@k."""
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0

    top_k = list(retrieved_ids)[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(top_k, start=1)
        if doc_id in relevant_set
    )

    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0