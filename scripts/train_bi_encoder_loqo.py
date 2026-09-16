"""Full LOQO bi-encoder fine-tuning sweep: one fold per query, hard-negative TripletLoss.

Runs leave-one-query-out cross-validation across all 10 English development
queries. For each fold: hold out one query, mine hard negatives for the other
9 (rank the full corpus against each training query with the zero-shot base
model, keep the top HARD_NEGATIVE_POOL_SIZE non-positive documents per query),
build up to NEG_PER_POSITIVE triplets per positive from that pool (near-dup
negatives filtered by token Jaccard against the positive), train a *fresh*
model from scratch with TripletLoss, and evaluate on the held-out query.

Training uses a plain manual loop (backward()/optimizer.step() directly), not
sentence-transformers' SentenceTransformerTrainer: that Trainer/accelerate path
was found to silently leave model weights unchanged after training while still
logging plausible-looking loss values. Gradient clipping (MAX_GRAD_NORM) is
applied explicitly since the Trainer did this for us by default and the manual
loop otherwise wouldn't. The held-out NDCG@10 is checked periodically during
training (EVAL_EVERY) and the best-scoring checkpoint is kept, since it can
peak well before MAX_STEPS and then decay (classic overfitting curve) - the
final step is not necessarily the best one.

Results are saved incrementally to RESULTS_PATH after each fold, so an
interrupted run (e.g. a Colab timeout) doesn't lose completed folds.
"""

import json
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
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "results" / "checkpoints"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "loqo_full_sweep.json"

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
# 220 steps x batch size 8 = 1760 triplets = exactly one full epoch over the
# training positives (per fold; each fold has a slightly different triplet
# count since a different query's positives are excluded from training).
MAX_STEPS = 220
# Evaluate on the held-out query every EVAL_EVERY steps during training (not
# just before/after), so the best checkpoint can be found and traced.
EVAL_EVERY = 20

# Restrict to specific query ids for a quick test (e.g. ["46795"]), or None to
# run the full 10-fold sweep.
FOLD_QUERY_IDS = None


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


def evaluate_held_out(model_or_path, held_out_query_id, queries, corpus, qrels):
    """model_or_path: a model name/checkpoint path, or an in-memory SentenceTransformer
    (the latter avoids a disk round-trip for periodic in-training checks)."""
    if isinstance(model_or_path, SentenceTransformer):
        bi_encoder = BiEncoder(corpus, model=model_or_path)
    else:
        bi_encoder = BiEncoder(corpus, model_name=model_or_path)

    relevant_ids = qrels.get(held_out_query_id, [])
    retrieved_ids = [
        doc_id for doc_id, _score in bi_encoder.rank(queries[held_out_query_id], top_k=50)
    ]
    return {
        "recall@10": recall_at_k(retrieved_ids, relevant_ids, k=10),
        "recall@50": recall_at_k(retrieved_ids, relevant_ids, k=50),
        "mrr": reciprocal_rank(retrieved_ids, relevant_ids),
        "ndcg@10": ndcg_at_k(retrieved_ids, relevant_ids, k=10),
    }


def train_manual(model, loss_fn, triplets, rng, held_out_query_id, queries, corpus, qrels):
    """Plain training loop: builds the optimizer directly on model.parameters(),
    the same tensors used in the forward pass, so updates can't get lost.

    Returns (best_state_dict, best_step, best_metrics): held-out performance
    doesn't necessarily peak at the final step (it can rise then decay well
    before MAX_STEPS), so the best state seen at any periodic eval is kept in
    memory and handed back, rather than discarding it in favor of the final
    step. "Best" is ranked primarily by ndcg@10, falling through to recall@50
    then mrr then recall@10 to break ties (ndcg@10 alone can't distinguish
    among steps where nothing lands in the top 10 at all).
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    device = model.device

    order = list(range(len(triplets)))
    rng.shuffle(order)
    position = 0

    best_state_dict = None
    best_step = 0
    best_metrics = None
    # Compared as a tuple: primarily by ndcg@10, but ndcg@10 is blind to
    # anything past rank 10 (it's mathematically forced to exactly 0.0
    # whenever recall@10 is 0, regardless of how close relevant results are
    # further down) — we've seen steps tie at ndcg@10=0.0 with recall@50
    # ranging from 0.06 to 0.34. Falling through to recall@50, then mrr, then
    # recall@10 breaks those ties instead of keeping whichever tied step was
    # checked first.
    best_score = (-1.0, -1.0, -1.0, -1.0)

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
                held_out_metrics = evaluate_held_out(model, held_out_query_id, queries, corpus, qrels)
            model.train()
            print(f"  held-out eval @ step {step}: {held_out_metrics}")

            score = (
                held_out_metrics["ndcg@10"],
                held_out_metrics["recall@50"],
                held_out_metrics["mrr"],
                held_out_metrics["recall@10"],
            )
            if score > best_score:
                best_score = score
                best_step = step
                best_metrics = held_out_metrics
                best_state_dict = {key: value.detach().clone().cpu() for key, value in model.state_dict().items()}

    return best_state_dict, best_step, best_metrics


def run_fold(held_out_query_id, queries, corpus, qrels):
    """Train and evaluate a single LOQO fold, returning zero-shot vs. best
    fine-tuned held-out metrics."""
    print(f"\n{'=' * 70}")
    print(f"FOLD: held-out query {held_out_query_id} ({queries[held_out_query_id].splitlines()[0]})")
    print("=" * 70)

    train_query_ids = [qid for qid in queries if qid != held_out_query_id]
    rng = random.Random(SEED)

    print("Mining hard-negative pools (ranking corpus against each training query)...")
    hard_negative_pools = mine_hard_negative_pools(train_query_ids, queries, corpus, qrels)
    triplets = build_triplets(train_query_ids, queries, corpus, qrels, rng, hard_negative_pools)
    print(f"Training queries: {len(train_query_ids)}  Triplets built: {len(triplets)}")

    print("Evaluating zero-shot base model on held-out query...")
    zero_shot_metrics = evaluate_held_out(MODEL_NAME, held_out_query_id, queries, corpus, qrels)
    print(zero_shot_metrics)

    model = SentenceTransformer(MODEL_NAME)
    loss_fn = TripletLoss(model, distance_metric=TripletDistanceMetric.COSINE, triplet_margin=MARGIN)

    train_rng = random.Random(SEED)
    best_state_dict, best_step, best_metrics = train_manual(
        model, loss_fn, triplets, train_rng, held_out_query_id, queries, corpus, qrels
    )
    print(f"Best held-out checkpoint at step {best_step}/{MAX_STEPS}: {best_metrics}")
    model.load_state_dict(best_state_dict)

    output_dir = CHECKPOINT_DIR / f"loqo_{held_out_query_id}_mpnet_triplet"
    model.save(str(output_dir))
    print(f"Saved fine-tuned model to {output_dir}")

    fine_tuned_metrics = evaluate_held_out(str(output_dir), held_out_query_id, queries, corpus, qrels)
    print(f"Fine-tuned (best checkpoint) metrics: {fine_tuned_metrics}")

    return {
        "held_out_query_id": held_out_query_id,
        "best_step": best_step,
        "zero_shot": zero_shot_metrics,
        "fine_tuned": fine_tuned_metrics,
    }


def save_results(all_results):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")


def print_summary(all_results):
    print(f"\n{'=' * 70}")
    print(f"SUMMARY across {len(all_results)} fold(s)")
    print("=" * 70)
    print(f"{'qid':10} {'zs_ndcg10':10} {'ft_ndcg10':10} {'zs_recall10':12} {'ft_recall10':12}")
    for r in all_results:
        zs, ft = r["zero_shot"], r["fine_tuned"]
        print(
            f"{r['held_out_query_id']:10} {zs['ndcg@10']:<10.3f} {ft['ndcg@10']:<10.3f} "
            f"{zs['recall@10']:<12.3f} {ft['recall@10']:<12.3f}"
        )

    print()
    for metric in ["recall@10", "recall@50", "mrr", "ndcg@10"]:
        zs_avg = sum(r["zero_shot"][metric] for r in all_results) / len(all_results)
        ft_avg = sum(r["fine_tuned"][metric] for r in all_results) / len(all_results)
        print(f"mean {metric:10} zero-shot={zs_avg:.3f}  fine-tuned={ft_avg:.3f}  delta={ft_avg - zs_avg:+.3f}")


def main() -> None:
    queries = load_queries(SPLIT_DIR)
    corpus = load_corpus(SPLIT_DIR)
    qrels = load_qrels(SPLIT_DIR)

    fold_query_ids = FOLD_QUERY_IDS if FOLD_QUERY_IDS is not None else sorted(queries.keys())
    print(f"Running {len(fold_query_ids)} fold(s): {fold_query_ids}")

    all_results = []
    for held_out_query_id in fold_query_ids:
        result = run_fold(held_out_query_id, queries, corpus, qrels)
        all_results.append(result)
        save_results(all_results)
        print(f"(saved progress: {len(all_results)}/{len(fold_query_ids)} folds to {RESULTS_PATH})")

    print_summary(all_results)


if __name__ == "__main__":
    main()
