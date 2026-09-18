"""Inspection report for the prepared Profile-Jobs-Ranked dataset.

Run after scripts/prepare_profile_jobs_ranked.py. Produces a human-readable
report (printed and saved to results/profile_jobs_ranked/inspection_report.md)
covering: dataset dimensions, split profile/pair counts, grade statistics,
candidates-per-profile distribution, example profiles with candidates sorted
by grade and by retrieval rank, examples of an obvious positive / plausible
hard negative / easy negative, and data-quality flags. Read-only - no
preprocessing, no triplet construction, no training.
"""

import random
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "profile_jobs_ranked" / "processed"
RESULTS_DIR = REPO_ROOT / "results" / "profile_jobs_ranked"
REPORT_PATH = RESULTS_DIR / "inspection_report.md"

SEED = 42
N_EXAMPLE_PROFILES = 3
SUB_SCORE_COLS = ["eligibility", "role_fit", "seniority_fit", "skill_fit", "location_fit", "comp_fit"]


def load_all() -> pd.DataFrame:
    frames = []
    for filename in ["train.parquet", "validation.parquet", "test.parquet"]:
        path = PROCESSED_DIR / filename
        if not path.exists():
            print(f"{path} not found - run scripts/prepare_profile_jobs_ranked.py first.")
            sys.exit(1)
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def fmt_pair(job_id, retrieval_rank, bucket, grade) -> str:
    return f"  job_id={job_id[:16]:<16}  rank={retrieval_rank:<5}  bucket={bucket}  grade={grade:>5.1f}"


def main() -> None:
    df = load_all()
    out = []

    def emit(line: str = "") -> None:
        print(line)
        out.append(line)

    emit("# Profile-Jobs-Ranked: inspection report\n")

    emit("## 1. Dataset dimensions\n")
    emit(f"- total rows (profile-job pairs): {len(df)}")
    emit(f"- unique profiles: {df['profile_id'].nunique()}")
    emit(f"- unique jobs: {df['job_id'].nunique()}")
    emit("")

    emit("## 2. Train/validation/test profile counts\n")
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        emit(f"- {split}: {sub['profile_id'].nunique()} profiles")
    emit("")

    emit("## 3. Pair counts per split\n")
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        emit(f"- {split}: {len(sub)} pairs")
    emit("")

    emit("## 4. Grade statistics\n")
    g = df["grade"]
    emit(f"- mean={g.mean():.2f}  median={g.median():.2f}  std={g.std():.2f}")
    quantiles = g.quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    emit("- quantiles: " + ", ".join(f"{q:.0%}={v:.1f}" for q, v in quantiles.items()))
    bands = pd.cut(g, bins=[-1, 20, 40, 60, 75, 90, 100],
                    labels=["0-20 Disqualified", "21-40 Poor", "41-60 Marginal", "61-75 Fair", "76-90 Strong", "91-100 Excellent"])
    emit("- band distribution:")
    for band, count in bands.value_counts().sort_index().items():
        emit(f"    {band}: {count} ({100 * count / len(df):.1f}%)")
    emit("")

    emit("## 5. Candidates per profile\n")
    per_profile = df.groupby("profile_id").size()
    emit(f"- mean={per_profile.mean():.2f}  min={per_profile.min()}  max={per_profile.max()}  median={per_profile.median():.0f}")
    emit(f"- profiles with < 25 candidates (thin market): {(per_profile < 25).sum()} ({100 * (per_profile < 25).mean():.1f}%)")
    emit("")

    emit("## 6. Example profiles: candidates sorted by grade vs. by retrieval rank\n")
    candidates_per_profile = per_profile[per_profile >= 10]
    rng = random.Random(SEED)
    example_pids = rng.sample(list(candidates_per_profile.index), min(N_EXAMPLE_PROFILES, len(candidates_per_profile)))
    for pid in example_pids:
        prof_rows = df[df["profile_id"] == pid]
        emit(f"### profile {pid}\n")
        emit("Profile text:")
        emit("```")
        emit(prof_rows.iloc[0]["profile_text"])
        emit("```")
        emit("\nCandidates sorted by grade (desc):")
        for _, row in prof_rows.sort_values("grade", ascending=False).iterrows():
            emit(fmt_pair(row["job_id"], row["retrieval_rank"], row["bucket"], row["grade"]))
        emit("\nCandidates sorted by retrieval_rank (asc):")
        for _, row in prof_rows.sort_values("retrieval_rank").iterrows():
            emit(fmt_pair(row["job_id"], row["retrieval_rank"], row["bucket"], row["grade"]))
        emit("")

    emit("## 7. Concrete positive / hard-negative / easy-negative examples\n")
    for pid in example_pids:
        prof_rows = df[df["profile_id"] == pid].sort_values("grade", ascending=False)
        best = prof_rows.iloc[0]
        mid_candidates = prof_rows[(prof_rows["grade"] < best["grade"] - 15) & (prof_rows["grade"] >= 30)]
        worst = prof_rows.iloc[-1]
        emit(f"### profile {pid}\n")
        positive_label = "obvious positive" if best["grade"] >= 61 else "best available (thin market - no Fair+ candidate exists)"
        emit(f"- {positive_label} (grade={best['grade']:.0f}): {best['job_text'].splitlines()[0]}")
        if len(mid_candidates) > 0:
            mid = mid_candidates.iloc[0]
            emit(f"- plausible hard negative (grade={mid['grade']:.0f}, rank={mid['retrieval_rank']}): {mid['job_text'].splitlines()[0]}")
        else:
            emit("- plausible hard negative: none found for this profile at the [30, best-15) gap")
        emit(f"- easy negative (grade={worst['grade']:.0f}): {worst['job_text'].splitlines()[0]}")
        emit("")

    emit("## 8. Data-quality flags\n")
    null_counts = df[["profile_text", "job_text"] + SUB_SCORE_COLS].isnull().sum()
    for col, n in null_counts.items():
        if n > 0:
            emit(f"- {col}: {n} nulls ({100 * n / len(df):.2f}%)")
    if null_counts.sum() == 0:
        emit("- no nulls found in profile_text/job_text/sub-score columns")

    inconsistent = (df["eligibility"] == 0) & (df["grade"] > 20)
    emit(f"- inconsistent sub-scores (eligibility=0 but grade>20): {inconsistent.sum()} ({100 * inconsistent.mean():.2f}%)")

    no_fair_match = per_profile.index[~per_profile.index.isin(df[df["grade"] >= 61]["profile_id"])]
    emit(f"- profiles with zero candidates graded >=61 (Fair+): {len(no_fair_match)} ({100 * len(no_fair_match) / len(per_profile):.1f}%)"
         " - relevant to any later positive-threshold choice")

    dupe_jobs_across_profiles = df.groupby("job_id")["profile_id"].nunique()
    emit(f"- jobs appearing for >1 profile: {(dupe_jobs_across_profiles > 1).sum()} of {len(dupe_jobs_across_profiles)} unique jobs"
         " (expected - candidates are drawn from a shared job pool, not a data quality issue by itself)")
    emit("")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(out), encoding="utf-8")
    print(f"\nSaved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
