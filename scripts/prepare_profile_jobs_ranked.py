"""Prepare the Profile-Jobs-Ranked dataset for later bi-encoder fine-tuning.

Preprocessing, splitting, and validation ONLY - no triplet construction, no
training. Pipeline:

1. Sample N_PROFILES profile_ids from the candidate shards (reading shards in
   order until the target count is reached, not all 125 - shards were checked
   to be cohort-representative, not grouped).
2. Split profile_ids 80/10/10 into train/val/test (fixed seed, profile-level -
   never split individual profile-job pairs).
3. Filter profiles.parquet and full_graded.parquet down to the sampled
   profile_ids (single bounded reads of the full files, immediately filtered -
   not held in memory beyond that).
4. Inner-join grades <-> candidates <-> profiles, printing row counts at each
   step so drops are explained.
5. Render deterministic profile_text / job_text for every row.
6. Run validation checks (fail loudly on any violation).
7. Write data/profile_jobs_ranked/processed/{train,validation,test}.parquet
   and results/profile_jobs_ranked/{splits/*.json, dataset_statistics.json}.
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from profile_jobs_ranked import (  # noqa: E402
    REPO_ID,
    REPO_TYPE,
    render_job,
    render_profile,
    sample_profile_ids_from_shards,
    split_profile_ids,
)
from huggingface_hub import hf_hub_download  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "profile_jobs_ranked" / "processed"
RESULTS_DIR = REPO_ROOT / "results" / "profile_jobs_ranked"
SPLITS_DIR = RESULTS_DIR / "splits"

# Dev-subset scale. Bump to 50_000, then 100_000+ once this is validated.
N_PROFILES = 10_000
SEED = 42
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1  # remainder (0.1) goes to test

RETRIEVAL_KEY_COLS = ["profile_id", "job_id", "retrieval_rank", "score", "bucket"]


def _shard_filename(shard_index: int) -> str:
    return f"shard_{shard_index:04d}.parquet"


def main() -> None:
    t0 = time.time()

    print(f"Sampling {N_PROFILES} profile_ids from candidate shards (seed={SEED})...")
    profile_ids, shard_indices = sample_profile_ids_from_shards(N_PROFILES, SEED)
    print(f"  sampled {len(profile_ids)} profile_ids from {len(shard_indices)} shard(s): {shard_indices}")

    train_ids, val_ids, test_ids = split_profile_ids(profile_ids, SEED, TRAIN_FRAC, VAL_FRAC)
    print(f"  split: train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        (SPLITS_DIR / f"{name}_profile_ids.json").write_text(json.dumps(sorted(ids), indent=2), encoding="utf-8")
    print(f"  saved split profile_ids to {SPLITS_DIR}")

    profile_id_set = set(profile_ids)
    profile_to_split = {pid: "train" for pid in train_ids}
    profile_to_split.update({pid: "val" for pid in val_ids})
    profile_to_split.update({pid: "test" for pid in test_ids})

    print("\nLoading candidates (reusing already-downloaded/cached shards)...")
    candidate_frames = []
    for shard_index in shard_indices:
        path = hf_hub_download(repo_id=REPO_ID, filename=_shard_filename(shard_index), repo_type=REPO_TYPE)
        shard_df = pq.read_table(path).to_pandas()
        candidate_frames.append(shard_df[shard_df["profile_id"].isin(profile_id_set)])
    candidates = pd.concat(candidate_frames, ignore_index=True)
    print(f"  candidates rows for sampled profiles: {len(candidates)}")

    print("\nLoading profiles.parquet (single bounded read, filtered immediately)...")
    profiles_path = hf_hub_download(repo_id=REPO_ID, filename="profiles.parquet", repo_type=REPO_TYPE)
    profiles_full = pq.read_table(profiles_path).to_pandas()
    print(f"  profiles.parquet total rows: {len(profiles_full)}")
    profiles = profiles_full[profiles_full["profile_id"].isin(profile_id_set)].copy()
    del profiles_full
    print(f"  profiles rows for sampled profiles: {len(profiles)}")

    print("\nLoading full_graded.parquet (single bounded read, filtered immediately)...")
    grades_path = hf_hub_download(repo_id=REPO_ID, filename="full_graded.parquet", repo_type=REPO_TYPE)
    grades_full = pq.read_table(grades_path).to_pandas()
    print(f"  full_graded.parquet total rows: {len(grades_full)}")
    grades = grades_full[grades_full["profile_id"].isin(profile_id_set)].copy()
    del grades_full
    print(f"  grades rows for sampled profiles: {len(grades)}")

    print("\nJoining grades <-> candidates <-> profiles...")
    grade_cols = ["profile_id", "job_id", "eligibility", "role_fit", "seniority_fit", "skill_fit", "location_fit", "comp_fit", "grade"]
    merged = candidates.merge(grades[grade_cols], on=["profile_id", "job_id"], how="inner")
    print(f"  candidates ({len(candidates)}) inner-join grades ({len(grades)}) -> {len(merged)} rows"
          f"  (dropped {len(candidates) - len(merged)} candidate rows with no grade,"
          f" {len(grades) - len(merged)} grade rows with no matching candidate)")

    merged = merged.merge(profiles[["profile_id", "seed_role", "profile_json"]], on="profile_id", how="inner")
    print(f"  -> join profiles -> {len(merged)} rows")

    print("\nRendering profile_text (cached per unique profile_id) and job_text...")
    profile_text_cache = {}
    for _, prow in profiles.iterrows():
        profile_text_cache[prow["profile_id"]] = render_profile(prow.to_dict())
    merged["profile_text"] = merged["profile_id"].map(profile_text_cache)
    merged["job_text"] = merged.apply(lambda row: render_job(row.to_dict()), axis=1)

    merged["retrieval_score"] = merged["score"]
    merged["split"] = merged["profile_id"].map(profile_to_split)

    final_cols = [
        "profile_id", "job_id", "profile_text", "job_text", "grade",
        "eligibility", "role_fit", "seniority_fit", "skill_fit", "location_fit", "comp_fit",
        "retrieval_rank", "retrieval_score", "bucket", "split",
    ]
    final = merged[final_cols].reset_index(drop=True)

    print("\nRunning validation checks...")
    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)
    assert not (train_set & val_set), "train/val profile_id overlap"
    assert not (train_set & test_set), "train/test profile_id overlap"
    assert not (val_set & test_set), "val/test profile_id overlap"
    print("  [ok] no profile overlap across splits")

    n_dupe_pairs = final.duplicated(subset=["profile_id", "job_id"]).sum()
    assert n_dupe_pairs == 0, f"{n_dupe_pairs} duplicate (profile_id, job_id) pairs in final output"
    print("  [ok] no duplicate (profile_id, job_id) pairs")

    assert final["profile_id"].notna().all() and final["job_id"].notna().all(), "null profile_id/job_id"
    print("  [ok] profile_id/job_id non-null")

    assert final["grade"].between(0, 100).all(), "grade outside [0, 100]"
    print("  [ok] grades within [0, 100]")

    empty_profile_text = (final["profile_text"].str.strip() == "").sum()
    empty_job_text = (final["job_text"].str.strip() == "").sum()
    assert empty_profile_text == 0, f"{empty_profile_text} rows with empty profile_text"
    assert empty_job_text == 0, f"{empty_job_text} rows with empty job_text"
    print("  [ok] profile_text/job_text non-empty")

    for name, expected_ids in [("train", train_set), ("val", val_set), ("test", test_set)]:
        actual_ids = set(final.loc[final["split"] == name, "profile_id"].unique())
        assert actual_ids <= expected_ids, f"{name} split contains profile_ids outside its assigned set"
    print("  [ok] each split's rows only contain their assigned profile_ids")

    print(f"\nWriting processed parquet files to {PROCESSED_DIR}...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    split_row_counts = {}
    split_profile_counts = {}
    for name in ["train", "val", "test"]:
        split_df = final[final["split"] == name]
        out_name = "validation" if name == "val" else name
        split_df.to_parquet(PROCESSED_DIR / f"{out_name}.parquet", index=False)
        split_row_counts[name] = len(split_df)
        split_profile_counts[name] = split_df["profile_id"].nunique()
        print(f"  {out_name}.parquet: {len(split_df)} rows, {split_df['profile_id'].nunique()} profiles")

    stats = {
        "n_profiles_sampled": len(profile_ids),
        "shard_indices_read": shard_indices,
        "seed": SEED,
        "split_row_counts": split_row_counts,
        "split_profile_counts": split_profile_counts,
        "total_rows": len(final),
        "grade_stats": {
            "mean": float(final["grade"].mean()),
            "median": float(final["grade"].median()),
            "min": float(final["grade"].min()),
            "max": float(final["grade"].max()),
            "std": float(final["grade"].std()),
        },
        "candidates_per_profile": {
            "mean": float(final.groupby("profile_id").size().mean()),
            "min": int(final.groupby("profile_id").size().min()),
            "max": int(final.groupby("profile_id").size().max()),
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (RESULTS_DIR / "dataset_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nSaved dataset statistics to {RESULTS_DIR / 'dataset_statistics.json'}")
    print(f"\nDone in {stats['elapsed_seconds']}s. Total rows: {stats['total_rows']}")


if __name__ == "__main__":
    main()
