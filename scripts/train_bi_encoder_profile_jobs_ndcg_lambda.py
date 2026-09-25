"""Profile-Jobs-Ranked fine-tuning with a true LambdaRank-style NDCG-aware loss.

Third and final experiment in the graded-relevance-aware objectives roadmap,
sibling to train_bi_encoder_profile_jobs_adaptive_margin.py and
train_bi_encoder_profile_jobs_weighted_pairwise.py. Same data split, same
validation-based checkpoint selection / single test evaluation protocol,
same resume mechanics - only the loss (and, necessarily, the batch unit)
changes.

Where weighted-pairwise weighted each pair by a STATIC function of the grade
gap alone, this experiment weights each pair by the actual |delta NDCG@10|
from swapping that pair's current rank positions within the profile's real
candidate pool - the standard LambdaRank formulation. Computing rank
position requires scoring a profile's ENTIRE candidate pool under the
model's current weights, not just two sampled candidates, so the batch unit
here is PROFILES (with their full candidate pool), not independent triplets.
build_triplets_from_grades is still used, but only to derive which profiles
are "eligible" (same population as the other two experiments: max grade
clears POSITIVE_THRESHOLD, has >=1 qualifying hard negative) - the actual
per-step training data is each eligible profile's full candidate pool from
train_df, grouped by profile_id.

Cost note: each step encodes roughly PROFILES_PER_BATCH * (1 + ~25) texts,
vs. the triplet scripts' BATCH_SIZE * 3 = 24 texts/step - about 4x more
forward-pass work per step. MAX_STEPS/EVAL_EVERY are kept identical to the
other two experiments anyway, to keep the validation-curve step axis
directly comparable, at the cost of a longer wall-clock run.

Reuses the exact linear-gain, 1/log2(rank+1)-discount NDCG convention
already established in graded_ndcg_at_k (src/evaluation.py) - same gain
function, same discount, just applied pairwise instead of summed over a
single ranked list.

This script is purely additive: it does not modify or replace the baseline,
adaptive-margin, or weighted-pairwise scripts, and writes to its own
separate checkpoint/results/resume paths.

Interruption recovery works exactly like the other Profile-Jobs-Ranked
scripts: a RESUME checkpoint is written to RESUME_CHECKPOINT_PATH after
every EVAL_EVERY-step validation check, and re-running this script picks it
back up automatically. Three distinct checkpoints exist here, same as the
others:
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
CHECKPOINT_DIR = REPO_ROOT / "results" / "checkpoints" / "profile_jobs_ranked_mpnet_ndcg_lambda"
RESULTS_PATH = REPO_ROOT / "results" / "profile_jobs_ranked" / "finetune_results_ndcg_lambda.json"
# Lives under the repo's own results/checkpoints/ so it survives a Colab
# runtime reset when the repo itself is checked out on Drive.
RESUME_CHECKPOINT_PATH = REPO_ROOT / "results" / "checkpoints" / "profile_jobs_ranked_mpnet_ndcg_lambda_resume.pt"

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
SEED = 42
# Unit is now PROFILES (each with its full candidate pool), not triplets -
# see module docstring's cost note.
PROFILES_PER_BATCH = 4
LEARNING_RATE = 1e-6
MAX_GRAD_NORM = 1.0
NEG_PER_POSITIVE = 4
POSITIVE_THRESHOLD = 61
NEGATIVE_GAP = 20
# Matches evaluate_profiles' graded_ndcg_at_k(..., k=10) - this loss directly
# targets the same metric the project reports.
NDCG_K = 10

MAX_STEPS = 2500
LOG_EVERY = 20
EVAL_EVERY = 250


def to_device(features, device):
    return {key: (value.to(device) if hasattr(value, "to") else value) for key, value in features.items()}


def lambda_pair_loss(anchor_emb, candidate_embs, grades, k=NDCG_K):
    """Vectorized true-LambdaRank pairwise loss for one profile's candidate
    pool: each pair's weight is |delta NDCG@k| from swapping that pair's
    CURRENT rank positions (computed from the model's live scores) - not a
    static function of the grade gap alone, unlike the weighted-pairwise
    experiment.

    anchor_emb: (dim,) profile embedding.
    candidate_embs: (n, dim) embeddings of every candidate in this profile's
    pool (not just a sampled positive/negative pair).
    grades: (n,) the same candidates' 0-100 grades.

    Returns (summed_loss, n_valid_pairs), or None if this profile's pool has
    no usable positive (idcg<=0) or no pair with differing grades - neither
    should actually occur for a profile that cleared build_triplets_from_grades'
    eligibility rule, but this mirrors that function's own skip behavior
    defensively.
    """
    scores = F.cosine_similarity(candidate_embs, anchor_emb.unsqueeze(0), dim=1)  # (n,)
    ranks = scores.argsort(descending=True).argsort() + 1  # (n,) current 1-indexed rank
    disc = torch.where(
        ranks <= k, 1.0 / torch.log2(ranks.float() + 1), torch.zeros_like(ranks, dtype=torch.float)
    )

    ideal_grades = grades.sort(descending=True).values[:k]
    ideal_disc = 1.0 / torch.log2(
        torch.arange(1, len(ideal_grades) + 1, dtype=torch.float, device=grades.device) + 1
    )
    idcg = (ideal_grades * ideal_disc).sum()
    if idcg.item() <= 0:
        return None

    gain_diff = grades.unsqueeze(1) - grades.unsqueeze(0)  # (n, n)
    disc_diff = disc.unsqueeze(1) - disc.unsqueeze(0)  # (n, n)
    delta_ndcg = (gain_diff.abs() * disc_diff.abs()) / idcg  # |delta NDCG@k| per pair

    score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)
    sign = torch.sign(gain_diff)  # +1/-1/0 - which side "should" score higher
    pair_loss = F.softplus(-sign * score_diff)  # RankNet loss per ordered pair

    weighted = delta_ndcg * pair_loss
    mask = gain_diff.triu(diagonal=1) != 0  # each unordered pair once, skip equal-grade pairs
    n_pairs = int(mask.sum().item())
    if n_pairs == 0:
        return None
    return weighted[mask].sum(), n_pairs


def _save_resume_checkpoint(
    path, step, model, optimizer, best_state_dict, best_step, best_metrics, best_score,
    validation_curve, order, position, rng, zero_shot_val, zero_shot_test,
):
    """Everything needed to continue this exact run from step+1: live model
    and optimizer state (not just the best-so-far model), both RNGs (Python's
    for profile sampling, torch's for dropout etc.), and the shuffled index
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


def train_manual(
    model, eligible_profile_ids, profile_groups, rng, validation_data, zero_shot_val, zero_shot_test, resume_state=None
):
    """Plain training loop: optimizer built directly on model.parameters(), the
    same tensors used in the forward pass. Each step samples PROFILES_PER_BATCH
    profile_ids, scores each one's FULL candidate pool, and computes the
    LambdaRank loss (lambda_pair_loss) per profile - summed across the batch
    and divided by the total pair count for the step's loss.

    Periodically evaluates on validation_data (never the test set) and keeps
    the best-scoring checkpoint in memory, ranked by
    (ndcg@10, recall@50, mrr, recall@10).

    If resume_state is given (loaded from RESUME_CHECKPOINT_PATH), training
    continues from resume_state["step"] + 1 with the exact model/optimizer
    weights, RNG states, and sampling position it had at that point.

    Returns (best_state_dict, best_step, best_metrics, validation_curve).
    """
    device = model.device

    if resume_state is not None:
        model.load_state_dict(resume_state["model_state_dict"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
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
        order = list(range(len(eligible_profile_ids)))
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
        while len(batch_indices) < PROFILES_PER_BATCH:
            if position >= len(order):
                rng.shuffle(order)
                position = 0
            batch_indices.append(order[position])
            position += 1
        batch_profile_ids = [eligible_profile_ids[i] for i in batch_indices]

        profile_losses = []
        total_pairs = 0
        for profile_id in batch_profile_ids:
            group = profile_groups[profile_id]
            anchor_text = group["profile_text"].iloc[0]
            candidate_texts = group["job_text"].tolist()
            grades = torch.tensor(group["grade"].tolist(), device=device, dtype=torch.float32)

            # Anchor + all candidates in one preprocess/forward call (instead of
            # two separate calls) - halves per-profile forward-pass overhead,
            # same embeddings either way since each text is processed independently.
            all_features = to_device(model.preprocess([anchor_text] + candidate_texts), device)
            all_embs = model(all_features)["sentence_embedding"]
            anchor_emb = all_embs[0]
            candidate_embs = all_embs[1:]

            result = lambda_pair_loss(anchor_emb, candidate_embs, grades)
            if result is None:
                continue
            profile_loss_sum, n_pairs = result
            profile_losses.append(profile_loss_sum)
            total_pairs += n_pairs

        if total_pairs == 0:
            # Shouldn't happen for eligible profiles (see lambda_pair_loss's
            # docstring), but skip cleanly rather than divide by zero.
            continue

        loss = sum(profile_losses) / total_pairs

        optimizer.zero_grad()
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

    print("\nDeriving eligible profiles (same population as the other two experiments)...")
    triplets = build_triplets_from_grades(
        train_df,
        seed=SEED,
        neg_per_positive=NEG_PER_POSITIVE,
        positive_threshold=POSITIVE_THRESHOLD,
        negative_gap=NEGATIVE_GAP,
    )
    eligible_profile_ids = sorted({t["profile_id"] for t in triplets})
    print(f"  eligible profiles: {len(eligible_profile_ids)} (of {train_df['profile_id'].nunique()} train profiles)")

    profile_groups = {
        profile_id: sub_df
        for profile_id, sub_df in train_df[train_df["profile_id"].isin(eligible_profile_ids)].groupby("profile_id")
    }

    resume_state = _load_resume_checkpoint(RESUME_CHECKPOINT_PATH)
    if resume_state is not None:
        print(f"Loaded resume checkpoint from step {resume_state['step']}. Continuing from step {resume_state['step'] + 1}.")
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
        f"LambdaRank NDCG@{NDCG_K} loss, profiles_per_batch={PROFILES_PER_BATCH}, lr={LEARNING_RATE})..."
    )
    model = SentenceTransformer(MODEL_NAME)
    train_rng = random.Random(SEED)

    best_state_dict, best_step, best_val_metrics, validation_curve = train_manual(
        model, eligible_profile_ids, profile_groups, train_rng, val_df, zero_shot_val, zero_shot_test, resume_state,
    )
    print(f"\nBest validation checkpoint at step {best_step}/{MAX_STEPS}: {best_val_metrics}")
    model.load_state_dict(best_state_dict)

    model.save(str(CHECKPOINT_DIR))
    print(f"Saved fine-tuned model to {CHECKPOINT_DIR}")

    print("\nEvaluating fine-tuned model on FULL test set (exactly once)...")
    fine_tuned_test = evaluate_profiles(str(CHECKPOINT_DIR), test_df)
    print(f"  fine-tuned on FULL test set: {fine_tuned_test}")

    results = {
        "config": {
            "model_name": MODEL_NAME,
            "seed": SEED,
            "profiles_per_batch": PROFILES_PER_BATCH,
            "learning_rate": LEARNING_RATE,
            "max_grad_norm": MAX_GRAD_NORM,
            "neg_per_positive": NEG_PER_POSITIVE,
            "positive_threshold": POSITIVE_THRESHOLD,
            "negative_gap": NEGATIVE_GAP,
            "ndcg_k": NDCG_K,
            "max_steps": MAX_STEPS,
            "eval_every": EVAL_EVERY,
        },
        "n_eligible_profiles": len(eligible_profile_ids),
        "best_step": best_step,
        "zero_shot": {"validation_full": zero_shot_val, "test_full": zero_shot_test},
        "fine_tuned": {"validation_full_at_best_step": best_val_metrics, "test_full": fine_tuned_test},
        "validation_curve": validation_curve,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results to {RESULTS_PATH}")

    if RESUME_CHECKPOINT_PATH.exists():
        RESUME_CHECKPOINT_PATH.unlink()
        print(f"Removed resume checkpoint {RESUME_CHECKPOINT_PATH} (run complete).")

    print("\n=== Comparison on FULL test set (evaluated exactly once) ===")
    print(f"{'metric':10} {'zero-shot':10} {'fine-tuned':10}")
    for metric in zero_shot_test:
        print(f"{metric:10} {zero_shot_test[metric]:<10.3f} {fine_tuned_test[metric]:<10.3f}")


if __name__ == "__main__":
    main()
