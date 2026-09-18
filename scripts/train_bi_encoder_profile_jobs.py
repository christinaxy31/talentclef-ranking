"""First Profile-Jobs-Ranked fine-tuning experiment: graded hard negatives,
validation-based checkpoint selection, evaluated on test exactly once.

Single train/val/test split (not leave-one-query-out like the TalentCLEF
scripts) - see results/profile_jobs_ranked/ for how that split was built.
Positives and hard negatives come directly from each profile's own graded
candidate pool (src/profile_jobs_ranked.py's build_triplets_from_grades):
positive = the profile's highest-graded candidate if it clears
POSITIVE_THRESHOLD; hard negatives = candidates at least NEGATIVE_GAP points
below it. No separate hard-negative mining step is needed here (unlike
TalentCLEF), since the candidates are already a pre-retrieved, plausible pool.

Checkpoint selection here legitimately uses a validation subsample (never the
test set) - unlike the earlier TalentCLEF LOQO setup, where the held-out
query stood in for both validation and test, which was a real leakage bug we
found and removed. Here validation and test are genuinely disjoint, separately
sampled profile sets, so periodically checking validation NDCG@10 during
training and keeping the best-scoring checkpoint is valid. The full test set
is evaluated exactly once, after the best-by-validation checkpoint is already
selected and frozen.

Training uses the same plain manual loop as scripts/train_bi_encoder_loqo.py
(backward()/optimizer.step() directly, explicit gradient clipping) for the
same reason documented there: sentence-transformers' SentenceTransformerTrainer
was found to silently leave weights unchanged after training.
"""

import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sentence_transformers import SentenceTransformer  # noqa: E402
from sentence_transformers.sentence_transformer.losses import (  # noqa: E402
    TripletDistanceMetric,
    TripletLoss,
)

from profile_jobs_ranked import build_triplets_from_grades, evaluate_profiles  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "profile_jobs_ranked" / "processed"
CHECKPOINT_DIR = REPO_ROOT / "results" / "checkpoints" / "profile_jobs_ranked_mpnet_triplet"
RESULTS_PATH = REPO_ROOT / "results" / "profile_jobs_ranked" / "finetune_results.json"

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
SEED = 42
BATCH_SIZE = 8
LEARNING_RATE = 1e-6
MARGIN = 0.2
MAX_GRAD_NORM = 1.0
NEG_PER_POSITIVE = 4
POSITIVE_THRESHOLD = 61
NEGATIVE_GAP = 20

# Deliberately generous rather than precisely tuned: with ~19,900 triplets
# available, 1000 steps x batch 8 covers ~40% of one pass. Validation-based
# checkpoint selection (not a fixed step count) is what actually picks the
# good checkpoint here, unlike the TalentCLEF LOQO recipe.
MAX_STEPS = 1000
LOG_EVERY = 20
EVAL_EVERY = 50
# Full validation set is 996 profiles; evaluating that at every one of ~20
# checkpoints would be slow. A fixed seeded subsample is enough to compare
# checkpoints against each other - only the FINAL test evaluation must use
# the complete, untouched set.
N_VALIDATION_EVAL_PROFILES = 200


def to_device(features, device):
    return {key: (value.to(device) if hasattr(value, "to") else value) for key, value in features.items()}


def train_manual(model, loss_fn, triplets, rng, validation_subset):
    """Plain training loop: optimizer built directly on model.parameters(), the
    same tensors used in the forward pass. Periodically evaluates on
    validation_subset (never the test set) and keeps the best-scoring
    checkpoint in memory, ranked by (ndcg@10, recall@50, mrr, recall@10) -
    ndcg@10 primary, falling through to the others to break ties.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    device = model.device

    order = list(range(len(triplets)))
    rng.shuffle(order)
    position = 0

    best_state_dict = None
    best_step = 0
    best_metrics = None
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

        if step % LOG_EVERY == 0 or step == MAX_STEPS:
            print(f"step {step}/{MAX_STEPS}  loss={loss.item():.4f}")

        if step % EVAL_EVERY == 0 or step == MAX_STEPS:
            model.eval()
            with torch.no_grad():
                val_metrics = evaluate_profiles(model, validation_subset)
            model.train()
            print(f"  validation eval @ step {step}: {val_metrics}")

            score = (val_metrics["ndcg@10"], val_metrics["recall@50"], val_metrics["mrr"], val_metrics["recall@10"])
            if score > best_score:
                best_score = score
                best_step = step
                best_metrics = val_metrics
                best_state_dict = {key: value.detach().clone().cpu() for key, value in model.state_dict().items()}

    return best_state_dict, best_step, best_metrics


def main() -> None:
    t0 = time.time()

    print("Loading processed train/validation/test parquet...")
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val_df = pd.read_parquet(PROCESSED_DIR / "validation.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    print(f"  train: {len(train_df)} rows, {train_df['profile_id'].nunique()} profiles")
    print(f"  validation: {len(val_df)} rows, {val_df['profile_id'].nunique()} profiles")
    print(f"  test: {len(test_df)} rows, {test_df['profile_id'].nunique()} profiles")

    print("\nBuilding triplets from train (graded hard negatives, no separate mining step)...")
    triplets = build_triplets_from_grades(
        train_df,
        seed=SEED,
        neg_per_positive=NEG_PER_POSITIVE,
        positive_threshold=POSITIVE_THRESHOLD,
        negative_gap=NEGATIVE_GAP,
    )
    print(f"  triplets built: {len(triplets)}")

    print("\nSampling fixed validation subset for periodic checks during training...")
    val_profile_ids = sorted(val_df["profile_id"].unique())
    subset_ids = random.Random(SEED).sample(val_profile_ids, min(N_VALIDATION_EVAL_PROFILES, len(val_profile_ids)))
    validation_subset = val_df[val_df["profile_id"].isin(subset_ids)]
    print(f"  validation subset: {len(validation_subset)} rows, {validation_subset['profile_id'].nunique()} profiles")

    print("\nEvaluating zero-shot base model...")
    zero_shot_val = evaluate_profiles(MODEL_NAME, validation_subset)
    print(f"  zero-shot on validation subset: {zero_shot_val}")
    zero_shot_test = evaluate_profiles(MODEL_NAME, test_df)
    print(f"  zero-shot on FULL test set: {zero_shot_test}")

    print(f"\nFine-tuning (fresh {MODEL_NAME}, up to {MAX_STEPS} steps, margin={MARGIN}, lr={LEARNING_RATE})...")
    model = SentenceTransformer(MODEL_NAME)
    loss_fn = TripletLoss(model, distance_metric=TripletDistanceMetric.COSINE, triplet_margin=MARGIN)
    train_rng = random.Random(SEED)

    best_state_dict, best_step, best_val_metrics = train_manual(model, loss_fn, triplets, train_rng, validation_subset)
    print(f"\nBest validation checkpoint at step {best_step}/{MAX_STEPS}: {best_val_metrics}")
    model.load_state_dict(best_state_dict)

    model.save(str(CHECKPOINT_DIR))
    print(f"Saved fine-tuned model to {CHECKPOINT_DIR}")

    # Full test set evaluated exactly once, here, after the checkpoint was
    # already selected using only the validation subset above.
    print("\nEvaluating fine-tuned model on FULL test set (exactly once)...")
    fine_tuned_test = evaluate_profiles(str(CHECKPOINT_DIR), test_df)
    print(f"  fine-tuned on FULL test set: {fine_tuned_test}")

    results = {
        "config": {
            "model_name": MODEL_NAME,
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "margin": MARGIN,
            "max_grad_norm": MAX_GRAD_NORM,
            "neg_per_positive": NEG_PER_POSITIVE,
            "positive_threshold": POSITIVE_THRESHOLD,
            "negative_gap": NEGATIVE_GAP,
            "max_steps": MAX_STEPS,
            "eval_every": EVAL_EVERY,
            "n_validation_eval_profiles": N_VALIDATION_EVAL_PROFILES,
        },
        "n_triplets": len(triplets),
        "best_step": best_step,
        "zero_shot": {"validation_subset": zero_shot_val, "test_full": zero_shot_test},
        "fine_tuned": {"validation_subset_at_best_step": best_val_metrics, "test_full": fine_tuned_test},
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results to {RESULTS_PATH}")

    print("\n=== Comparison on FULL test set (evaluated exactly once) ===")
    print(f"{'metric':10} {'zero-shot':10} {'fine-tuned':10}")
    for metric in zero_shot_test:
        print(f"{metric:10} {zero_shot_test[metric]:<10.3f} {fine_tuned_test[metric]:<10.3f}")


if __name__ == "__main__":
    main()
