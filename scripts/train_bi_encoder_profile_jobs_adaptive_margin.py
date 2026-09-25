"""Profile-Jobs-Ranked fine-tuning with an adaptive-margin triplet loss.

Second experiment in the graded-relevance-aware objectives roadmap (see
README "Next Experiments"). Everything is identical to
scripts/train_bi_encoder_profile_jobs.py (same data split, same
build_triplets_from_grades rule, same manual training loop, same
hyperparameter defaults, same validation-based checkpoint selection / single
test evaluation protocol, same interruption-recovery scheme) EXCEPT the loss:
instead of one fixed margin for every triplet, the required anchor-positive
vs. anchor-negative separation scales with how far apart the positive and
negative grades actually are.

The library's TripletLoss bakes a single scalar margin in at construction
time, so it can't take a per-triplet margin - adaptive_triplet_loss below
reimplements the same forward pass (cosine distance, relu(pos - neg +
margin), mean) but with margin as a per-batch tensor instead of a constant.

This script is purely additive: it does not modify or replace
train_bi_encoder_profile_jobs.py's fixed-margin baseline, and writes to its
own separate checkpoint/results/resume paths so both experiments can be
compared side by side (and interrupted/resumed independently).

Interruption recovery (for long Colab runs that can disconnect mid-training):
a RESUME checkpoint is written to RESUME_CHECKPOINT_PATH after every
EVAL_EVERY-step validation check, and re-running this script picks it back up
automatically. This is a separate concept from the two other checkpoints this
script produces:
  - RESUME checkpoint (RESUME_CHECKPOINT_PATH): everything needed to continue
    this exact run from step+1 after an interruption - live (not best) model
    weights, optimizer state, RNG states, sampling position, and bookkeeping.
    Deleted once a run completes successfully.
  - best validation checkpoint (best_state_dict, in memory during training):
    the model snapshot with the highest validation NDCG@10 seen so far.
  - final saved model (CHECKPOINT_DIR): best_state_dict written to disk once
    training finishes, then evaluated on the test set exactly once.
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sentence_transformers import SentenceTransformer  # noqa: E402

from profile_jobs_ranked import build_triplets_from_grades, evaluate_profiles  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "profile_jobs_ranked" / "processed"
CHECKPOINT_DIR = REPO_ROOT / "results" / "checkpoints" / "profile_jobs_ranked_mpnet_adaptive_margin"
RESULTS_PATH = REPO_ROOT / "results" / "profile_jobs_ranked" / "finetune_results_adaptive_margin.json"
# Lives under the repo's own results/checkpoints/ so it survives a Colab
# runtime reset when the repo itself is checked out on Drive (do not assume
# /content/... is persistent). Deliberately a different file from the
# fixed-margin baseline's resume checkpoint, so the two experiments can never
# resume from each other's state.
RESUME_CHECKPOINT_PATH = (
    REPO_ROOT / "results" / "checkpoints" / "profile_jobs_ranked_mpnet_adaptive_margin_resume.pt"
)

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
SEED = 42
BATCH_SIZE = 8
LEARNING_RATE = 1e-6
MAX_GRAD_NORM = 1.0
NEG_PER_POSITIVE = 4
POSITIVE_THRESHOLD = 61
NEGATIVE_GAP = 20

# Grade gap == NEGATIVE_GAP (the smallest gap build_triplets_from_grades ever
# produces) maps to MIN_MARGIN, so those triplets get the exact same margin
# as the fixed-margin baseline - keeping the two experiments comparable at
# the floor. Grade gap == 100 (positive=100, negative=0) maps to MAX_MARGIN.
MIN_MARGIN = 0.2
MAX_MARGIN = 0.4

MAX_STEPS = 2500
LOG_EVERY = 20
# Also the cadence at which the RESUME checkpoint is refreshed (see module
# docstring).
EVAL_EVERY = 250


def to_device(features, device):
    return {key: (value.to(device) if hasattr(value, "to") else value) for key, value in features.items()}


def adaptive_margin(gaps: torch.Tensor) -> torch.Tensor:
    """Linear interpolation between MIN_MARGIN and MAX_MARGIN over the grade
    gap range [NEGATIVE_GAP, 100], clamped outside that range."""
    frac = ((gaps - NEGATIVE_GAP) / (100 - NEGATIVE_GAP)).clamp(0.0, 1.0)
    return MIN_MARGIN + (MAX_MARGIN - MIN_MARGIN) * frac


def adaptive_triplet_loss(model, anchor_features, positive_features, negative_features, gaps):
    """Same forward pass as sentence-transformers' TripletLoss (cosine
    distance, relu(dist_pos - dist_neg + margin), mean) but margin is a
    per-triplet tensor derived from the grade gap instead of one constant."""
    anchor_emb = model(anchor_features)["sentence_embedding"]
    positive_emb = model(positive_features)["sentence_embedding"]
    negative_emb = model(negative_features)["sentence_embedding"]

    distance_pos = 1 - F.cosine_similarity(anchor_emb, positive_emb)
    distance_neg = 1 - F.cosine_similarity(anchor_emb, negative_emb)
    margin = adaptive_margin(gaps)
    return F.relu(distance_pos - distance_neg + margin).mean()


def _save_resume_checkpoint(
    path, step, model, optimizer, best_state_dict, best_step, best_metrics, best_score,
    validation_curve, order, position, rng, zero_shot_val, zero_shot_test,
):
    """Everything needed to continue this exact run from step+1: live model
    and optimizer state (not just the best-so-far model), both RNGs (Python's
    for triplet sampling, torch's for dropout etc.), and the shuffled index
    order/position so the next sampled batch is unchanged by the interruption.
    Written atomically (temp file + os.replace) so a crash mid-write can't
    leave a corrupted resume checkpoint behind.
    """
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_state_dict": best_state_dict,
        "best_step": best_step,
        "best_metrics": best_metrics,
        "best_score": best_score,
        "validation_curve": validation_curve,
        "order": order,
        "position": position,
        "python_random_state": rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "zero_shot_val": zero_shot_val,
        "zero_shot_test": zero_shot_test,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def _load_resume_checkpoint(path):
    if not path.exists():
        return None
    # weights_only=False: this checkpoint bundles plain Python objects (RNG
    # states, validation_curve, ints) alongside tensors, not just weights.
    return torch.load(path, map_location="cpu", weights_only=False)


def train_manual(model, triplets, rng, validation_data, zero_shot_val, zero_shot_test, resume_state=None):
    """Plain training loop: optimizer built directly on model.parameters(), the
    same tensors used in the forward pass. Periodically evaluates on
    validation_data (never the test set) and keeps the best-scoring
    checkpoint in memory, ranked by (ndcg@10, recall@50, mrr, recall@10) -
    ndcg@10 primary, falling through to the others to break ties.

    If resume_state is given (loaded from RESUME_CHECKPOINT_PATH), training
    continues from resume_state["step"] + 1 with the exact model/optimizer
    weights, RNG states, and sampling position it had at that point - not a
    fresh restart. Otherwise this behaves exactly as before.

    Returns (best_state_dict, best_step, best_metrics, validation_curve) -
    validation_curve is every {step, loss, **val_metrics} checked along the
    way (not just the best one), for plotting validation NDCG@10 vs. step.
    """
    device = model.device

    if resume_state is not None:
        model.load_state_dict(resume_state["model_state_dict"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        # torch.load(map_location="cpu") puts optimizer state tensors on CPU
        # regardless of where they were saved from - move them back onto the
        # live model's device before training continues.
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)

        order = resume_state["order"]
        position = resume_state["position"]
        best_state_dict = resume_state["best_state_dict"]
        best_step = resume_state["best_step"]
        best_metrics = resume_state["best_metrics"]
        best_score = tuple(resume_state["best_score"])
        validation_curve = resume_state["validation_curve"]

        rng.setstate(resume_state["python_random_state"])
        torch.set_rng_state(resume_state["torch_rng_state"])
        if torch.cuda.is_available() and resume_state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng_state_all"])

        start_step = resume_state["step"] + 1
    else:
        order = list(range(len(triplets)))
        rng.shuffle(order)
        position = 0

        best_state_dict = None
        best_step = 0
        best_metrics = None
        best_score = (-1.0, -1.0, -1.0, -1.0)
        validation_curve = []

        start_step = 1

    for step in range(start_step, MAX_STEPS + 1):
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
        gaps = torch.tensor(
            [t["positive_grade"] - t["negative_grade"] for t in batch], device=device, dtype=torch.float32
        )

        optimizer.zero_grad()
        loss = adaptive_triplet_loss(model, anchor_features, positive_features, negative_features, gaps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()

        if step % LOG_EVERY == 0 or step == MAX_STEPS:
            print(f"step {step}/{MAX_STEPS}  loss={loss.item():.4f}")

        if step % EVAL_EVERY == 0 or step == MAX_STEPS:
            model.eval()
            with torch.no_grad():
                val_metrics = evaluate_profiles(model, validation_data)
            model.train()
            print(f"  validation eval @ step {step}: {val_metrics}")
            validation_curve.append({"step": step, "loss": loss.item(), **val_metrics})

            score = (val_metrics["ndcg@10"], val_metrics["recall@50"], val_metrics["mrr"], val_metrics["recall@10"])
            if score > best_score:
                best_score = score
                best_step = step
                best_metrics = val_metrics
                best_state_dict = {key: value.detach().clone().cpu() for key, value in model.state_dict().items()}

            _save_resume_checkpoint(
                RESUME_CHECKPOINT_PATH, step, model, optimizer,
                best_state_dict, best_step, best_metrics, best_score, validation_curve,
                order, position, rng, zero_shot_val, zero_shot_test,
            )
            print(f"  saved resume checkpoint at step {step} -> {RESUME_CHECKPOINT_PATH}")

    return best_state_dict, best_step, best_metrics, validation_curve


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

    resume_state = _load_resume_checkpoint(RESUME_CHECKPOINT_PATH)
    if resume_state is not None:
        print(f"Loaded resume checkpoint from step {resume_state['step']}. Continuing from step {resume_state['step'] + 1}.")
        # Zero-shot is a fixed baseline computed once and never touched by
        # training - reusing the saved values avoids redoing this work every
        # time a long run is resumed after a Colab interruption.
        zero_shot_val = resume_state["zero_shot_val"]
        zero_shot_test = resume_state["zero_shot_test"]
        print(f"  (reusing zero-shot results from the resume checkpoint) validation: {zero_shot_val}  test: {zero_shot_test}")
    else:
        print("No resume checkpoint found. Starting from step 1.")
        print("\nEvaluating zero-shot base model...")
        zero_shot_val = evaluate_profiles(MODEL_NAME, val_df)
        print(f"  zero-shot on FULL validation set: {zero_shot_val}")
        zero_shot_test = evaluate_profiles(MODEL_NAME, test_df)
        print(f"  zero-shot on FULL test set: {zero_shot_test}")

    print(
        f"\nFine-tuning (fresh {MODEL_NAME}, up to {MAX_STEPS} steps, "
        f"margin={MIN_MARGIN}-{MAX_MARGIN}, lr={LEARNING_RATE})..."
    )
    model = SentenceTransformer(MODEL_NAME)
    train_rng = random.Random(SEED)

    best_state_dict, best_step, best_val_metrics, validation_curve = train_manual(
        model, triplets, train_rng, val_df, zero_shot_val, zero_shot_test, resume_state,
    )
    print(f"\nBest validation checkpoint at step {best_step}/{MAX_STEPS}: {best_val_metrics}")
    model.load_state_dict(best_state_dict)

    model.save(str(CHECKPOINT_DIR))
    print(f"Saved fine-tuned model to {CHECKPOINT_DIR}")

    # Full test set evaluated exactly once, here, after the checkpoint was
    # already selected using only the validation set above. The zero-shot
    # test baseline (above) is a fixed reference point only - it never
    # influences checkpoint selection or training.
    print("\nEvaluating fine-tuned model on FULL test set (exactly once)...")
    fine_tuned_test = evaluate_profiles(str(CHECKPOINT_DIR), test_df)
    print(f"  fine-tuned on FULL test set: {fine_tuned_test}")

    results = {
        "config": {
            "model_name": MODEL_NAME,
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "min_margin": MIN_MARGIN,
            "max_margin": MAX_MARGIN,
            "max_grad_norm": MAX_GRAD_NORM,
            "neg_per_positive": NEG_PER_POSITIVE,
            "positive_threshold": POSITIVE_THRESHOLD,
            "negative_gap": NEGATIVE_GAP,
            "max_steps": MAX_STEPS,
            "eval_every": EVAL_EVERY,
        },
        "n_triplets": len(triplets),
        "best_step": best_step,
        "zero_shot": {"validation_full": zero_shot_val, "test_full": zero_shot_test},
        "fine_tuned": {"validation_full_at_best_step": best_val_metrics, "test_full": fine_tuned_test},
        # Every periodic checkpoint's {step, loss, recall@10, recall@50, mrr,
        # ndcg@10} on the full validation set, in order - e.g. for plotting
        # validation NDCG@10 vs. training step.
        "validation_curve": validation_curve,
        # Note: on a resumed run this only covers the final (resumed) segment
        # of wall-clock time, not the total across interruptions.
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results to {RESULTS_PATH}")

    # Only remove the resume checkpoint after every final output above was
    # written successfully, so a later invocation starts a fresh run instead
    # of mistakenly resuming a run that already completed.
    if RESUME_CHECKPOINT_PATH.exists():
        RESUME_CHECKPOINT_PATH.unlink()
        print(f"Removed resume checkpoint {RESUME_CHECKPOINT_PATH} (run complete).")

    print("\n=== Comparison on FULL test set (evaluated exactly once) ===")
    print(f"{'metric':10} {'zero-shot':10} {'fine-tuned':10}")
    for metric in zero_shot_test:
        print(f"{metric:10} {zero_shot_test[metric]:<10.3f} {fine_tuned_test[metric]:<10.3f}")


if __name__ == "__main__":
    main()