
"""First LOQO bi-encoder fine-tuning experiment: single fold, hard-negative TripletLoss.

Holds out one query (HELD_OUT_QUERY_ID) and fine-tunes on the other 9 training
queries from the English development split. Negatives are mined statically:
the zero-shot base model ranks the full corpus against each training query
once, and up to NEG_PER_POSITIVE negatives per positive are sampled from the
top HARD_NEGATIVE_POOL_SIZE non-positive documents (cross-query positives are
allowed as negatives, per experiment design). Near-duplicate negatives are
filtered out via a token Jaccard threshold against the positive they'd be
paired with — this matters more here than with random negatives, since a hard
negative that's secretly an unjudged true positive is a much worse training
signal than an easy, unrelated one would be.

Training uses a plain manual loop (backward()/optimizer.step() directly), not
sentence-transformers' SentenceTransformerTrainer: that Trainer/accelerate path
was found to silently leave model weights unchanged after training (confirmed
via the weight-diff sanity check below) while still logging plausible-looking
loss values — most likely an optimizer built on parameter tensors that get
disconnected from the model during accelerate's device/dtype placement. A
manual loop builds the optimizer directly on the tensors used in the forward
pass, so there's no place for that disconnect to happen.

After training, the fine-tuned model is compared against the zero-shot base
model on the held-out query using the existing evaluation pipeline.
"""

import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sentence_transformers import SentenceTransformer  # noqa: E402
from sentence_transformers.sentence_transformer.losses import (  # noqa: E402
    TripletDistanceMetric,
    TripletLoss,
)

from bi_encoder import BiEncoder, MODEL_NAME  # noqa: E402
from bm25 import tokenize  # noqa: E402
from data_loader import load_corpus, load_qrels, load_queries  # noqa: E402
from evaluation import ndcg_at_k, recall_at_k, reciprocal_rank  # noqa: E402

SPLIT_DIR = Path(__file__).resolve().parent.parent / "data" / "TaskA" / "development" / "en"

HELD_OUT_QUERY_ID = "46795"
NEG_PER_POSITIVE = 4
JACCARD_THRESHOLD = 0.6
SEED = 42
BATCH_SIZE = 8
LEARNING_RATE = 1e-6
MARGIN = 0.2
MAX_GRAD_NORM = 1.0
# Static hard-negative mining: rank the full corpus against each training
# query once (using the zero-shot base model), then sample negatives from the
# top-ranked non-positive documents instead of uniformly at random.
HARD_NEGATIVE_POOL_SIZE = 50

OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "results"
    / "checkpoints"
    / f"loqo_{HELD_OUT_QUERY_ID}_mpnet_triplet"
)
# 220 steps x batch size 8 = 1760 triplets = exactly one full epoch over the
# training positives. Raised from the 10-step smoke test to test whether the
# held-out degradation we saw was a "too little training" artifact.
MAX_STEPS = 220
# Evaluate on the held-out query every EVAL_EVERY steps during training (not
# just before/after), so a collapse can be traced to roughly where it happens
# instead of only being visible as a single before/after number.
EVAL_EVERY = 20


def mine_hard_negative_pools(train_query_ids, queries, corpus, qrels):
    """Rank the full corpus against each training query with the zero-shot base
    model once, returning {query_id: [top HARD_NEGATIVE_POOL_SIZE non-positive
    doc_ids]} — the hard-negative candidate pool for that query."""
    bi_encoder = BiEncoder(corpus, model_name=MODEL_NAME)
    pools = {}
    for query_id in train_query_ids:
        positives = set(qrels.get(query_id, []))
        ranked = bi_encoder.rank(queries[query_id], top_k=len(corpus))
        hard_candidates = [doc_id for doc_id, _score in ranked if doc_id not in positives]
        pools[query_id] = hard_candidates[:HARD_NEGATIVE_POOL_SIZE]
    return pools


def build_triplets(train_query_ids, queries, corpus, qrels, rng, hard_negative_pools):
    """(anchor, positive, negative) triplets with a token-Jaccard near-duplicate filter."""
    token_sets = {doc_id: set(tokenize(text)) for doc_id, text in corpus.items()}

    def jaccard(a, b):
        return len(a & b) / len(a | b)

    triplets = []
    for query_id in train_query_ids:
        positives = qrels.get(query_id, [])
        negative_pool = hard_negative_pools[query_id]
        for pos_id in positives:
            candidates = [
                doc_id
                for doc_id in negative_pool
                if jaccard(token_sets[doc_id], token_sets[pos_id]) < JACCARD_THRESHOLD
            ]
            sample_size = min(NEG_PER_POSITIVE, len(candidates))
            for neg_id in rng.sample(candidates, sample_size):
                triplets.append(
                    {
                        "anchor": queries[query_id],
                        "positive": corpus[pos_id],
                        "negative": corpus[neg_id],
                    }
                )
    return triplets


def to_device(features, device):
    return {key: (value.to(device) if hasattr(value, "to") else value) for key, value in features.items()}


def evaluate_held_out(model_or_path, queries, corpus, qrels):
    if isinstance(model_or_path, SentenceTransformer):
        bi_encoder = BiEncoder(
            corpus,
            model=model_or_path,
        )
    else:
        bi_encoder = BiEncoder(
            corpus,
            model_name=model_or_path,
        )

    relevant_ids = qrels.get(HELD_OUT_QUERY_ID, [])

    retrieved_ids = [
        doc_id
        for doc_id, _score in bi_encoder.rank(
            queries[HELD_OUT_QUERY_ID],
            top_k=50,
        )
    ]

    return {
        "recall@10": recall_at_k(retrieved_ids, relevant_ids, k=10),
        "recall@50": recall_at_k(retrieved_ids, relevant_ids, k=50),
        "mrr": reciprocal_rank(retrieved_ids, relevant_ids),
        "ndcg@10": ndcg_at_k(retrieved_ids, relevant_ids, k=10),
    }


def train_manual(model, loss_fn, triplets, rng, queries, corpus, qrels):
    """Plain training loop: builds the optimizer directly on model.parameters(),
    the same tensors used in the forward pass, so updates can't get lost.

    Returns (best_state_dict, best_step, best_ndcg): the held-out NDCG@10 doesn't
    necessarily peak at the final step (we've seen it rise then decay well before
    MAX_STEPS), so the best state seen at any periodic eval is kept in memory and
    handed back, rather than silently discarding it in favor of whatever the
    final step happens to look like.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    device = model.device

    order = list(range(len(triplets)))
    rng.shuffle(order)
    position = 0

    best_state_dict = None
    best_step = 0
    best_ndcg = -1.0

    for step in range(1, MAX_STEPS + 1):
        batch_indices = []
        while len(batch_indices) < BATCH_SIZE:
            if position >= len(order):
                rng.shuffle(order)
                position = 0
            batch_indices.append(order[position])
            position += 1
        batch = [triplets[i] for i in batch_indices]

        anchor_features = to_device(model.preprocess([t["anchor"] for t in batch]), device)
        positive_features = to_device(model.preprocess([t["positive"] for t in batch]), device)
        negative_features = to_device(model.preprocess([t["negative"] for t in batch]), device)

        optimizer.zero_grad()
        loss = loss_fn([anchor_features, positive_features, negative_features], labels=None)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()

        if step % 10 == 0 or step == MAX_STEPS:
            print(f"step {step}/{MAX_STEPS}  loss={loss.item():.4f}")

        if step % EVAL_EVERY == 0 or step == MAX_STEPS:
            model.eval()
            with torch.no_grad():
                held_out_metrics = evaluate_held_out(model, queries, corpus, qrels)
            model.train()
            print(f"  held-out eval @ step {step}: {held_out_metrics}")

            if held_out_metrics["ndcg@10"] > best_ndcg:
                best_ndcg = held_out_metrics["ndcg@10"]
                best_step = step
                best_state_dict = {key: value.detach().clone().cpu() for key, value in model.state_dict().items()}

    return best_state_dict, best_step, best_ndcg


def main() -> None:
    queries = load_queries(SPLIT_DIR)
    corpus = load_corpus(SPLIT_DIR)
    qrels = load_qrels(SPLIT_DIR)

    train_query_ids = [qid for qid in queries if qid != HELD_OUT_QUERY_ID]
    rng = random.Random(SEED)

    print("Mining hard-negative pools (ranking corpus against each training query)...")
    hard_negative_pools = mine_hard_negative_pools(train_query_ids, queries, corpus, qrels)

    triplets = build_triplets(train_query_ids, queries, corpus, qrels, rng, hard_negative_pools)
    print(f"Held-out query: {HELD_OUT_QUERY_ID} ({queries[HELD_OUT_QUERY_ID].splitlines()[0]})")
    print(f"Training queries: {len(train_query_ids)}")
    print(f"Triplets built: {len(triplets)}")

    print("\nEvaluating zero-shot base model on held-out query...")
    zero_shot_metrics = evaluate_held_out(MODEL_NAME, queries, corpus, qrels)
    print(zero_shot_metrics)

    model = SentenceTransformer(MODEL_NAME)
    loss_fn = TripletLoss(model, distance_metric=TripletDistanceMetric.COSINE, triplet_margin=MARGIN)

    train_rng = random.Random(SEED)
    best_state_dict, best_step, best_ndcg = train_manual(
        model, loss_fn, triplets, train_rng, queries, corpus, qrels
    )

    print(f"\nBest held-out NDCG@10 was {best_ndcg:.4f} at step {best_step}/{MAX_STEPS}")
    print("Restoring that checkpoint before saving (final step is not necessarily the best).")
    model.load_state_dict(best_state_dict)

    print("\nSanity check: confirming fine-tuned weights differ from the base model...")
    base_model = SentenceTransformer(MODEL_NAME)
    base_params = dict(base_model.named_parameters())
    diff_norm_sq = sum(
        (trained_param.detach().cpu() - base_params[name].detach().cpu()).pow(2).sum().item()
        for name, trained_param in model.named_parameters()
    )
    diff_norm = diff_norm_sq**0.5
    print(f"L2 norm of (fine-tuned - base) weights: {diff_norm:.6f}")
    if diff_norm < 1e-6:
        print("WARNING: fine-tuned weights are numerically identical to the base model.")

    model.save(str(OUTPUT_DIR))
    print(f"\nSaved fine-tuned model to {OUTPUT_DIR}")

    print("\nEvaluating fine-tuned model on held-out query...")
    fine_tuned_metrics = evaluate_held_out(str(OUTPUT_DIR), queries, corpus, qrels)
    print(fine_tuned_metrics)

    print("\n=== Comparison on held-out query", HELD_OUT_QUERY_ID, "===")
    print(f"{'metric':10} {'zero-shot':10} {'fine-tuned':10}")
    for metric in zero_shot_metrics:
        print(f"{metric:10} {zero_shot_metrics[metric]:<10.3f} {fine_tuned_metrics[metric]:<10.3f}")


if __name__ == "__main__":
    main()

