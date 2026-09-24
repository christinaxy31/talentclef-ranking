"""Reusable pieces for the Profile-Jobs-Ranked dataset (huggingface.co/datasets/akzaidan/Profile-Jobs-Ranked).

Real schema (verified by downloading and inspecting the actual parquet files,
not assumed):

  profiles.parquet: profile_id, source, cohort, job_family, seniority, country,
      seed_role, profile_json (a JSON string; see render_profile for its shape)

  shard_0000..0124.parquet (candidates): profile_id, job_id, retrieval_rank,
      score, bucket, title, company, city, state, country, is_remote,
      pay_range_lower, pay_range_upper, ai_salary, expected_experience_years,
      ai_summary, skills (JSON array string), desc, query_title, query_skills

  full_graded.parquet (grades): profile_id, job_id, retrieval_rank, score,
      bucket, eligibility, role_fit, seniority_fit, skill_fit, location_fit,
      comp_fit, grade
"""

import json
import random
import statistics
from typing import Dict, List, Tuple

from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

from evaluation import graded_ndcg_at_k, reciprocal_rank, recall_at_k

REPO_ID = "akzaidan/Profile-Jobs-Ranked"
REPO_TYPE = "dataset"
NUM_SHARDS = 125

# Fields never included in retrieval text. ethnicity/legal_status/
# sponsorship_needed/bio match the dataset's own whitelist renderer (excluded
# from the LLM grading prompt per its dataset card); age is added on top since
# it's a protected employment attribute the grading exclusion list doesn't
# happen to mention.
PROTECTED_USER_FIELDS = {"ethnicity", "legal_status", "sponsorship_needed", "bio", "age"}

# A candidate with grade >= this is "positive"/"relevant" - used both to pick
# the anchor positive for triplet construction and as the binary relevance
# threshold for recall_at_k/reciprocal_rank.
POSITIVE_GRADE_THRESHOLD = 61
# A hard negative must be at least this many grade points below the positive.
HARD_NEGATIVE_GAP = 20


def _present(value) -> bool:
    """True if value is a usable (non-missing, non-empty) scalar. Handles
    pandas' habit of representing a missing string column as float NaN after
    a parquet -> DataFrame read - NaN is truthy in plain Python (`bool(nan)`
    is True), so a plain `if value:` check silently lets it through."""
    if value is None:
        return False
    if isinstance(value, float) and value != value:  # NaN is the only float where x != x
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def render_profile(profile_row: Dict) -> str:
    """Deterministic text from a profiles.parquet row (profile_json parsed
    internally). Never includes PROTECTED_USER_FIELDS."""
    profile_json = profile_row["profile_json"]
    data = json.loads(profile_json) if isinstance(profile_json, str) else profile_json
    user = data.get("user", {}) or {}

    lines = []

    seed_role = profile_row.get("seed_role")
    if _present(seed_role):
        lines.append(f"Target role: {seed_role}")

    career_interests = user.get("career_interests") or []
    if career_interests:
        lines.append("Career interests: " + ", ".join(career_interests))

    experience_years = user.get("experience_years")
    if experience_years is not None:
        lines.append(f"Experience: {experience_years} years")

    skills = user.get("skills") or []
    if skills:
        # Checked as consistently plain strings across a 30k-profile sample, but
        # job-side skills turned out to have a dict-shaped variant in some rows
        # (see render_job) - handle both defensively here too, cheaply.
        skill_names = [s["name"] if isinstance(s, dict) else s for s in skills]
        skill_names = [s for s in skill_names if s]
        if skill_names:
            lines.append("Skills: " + ", ".join(skill_names))

    for exp in data.get("experience") or []:
        title = exp.get("title")
        company = exp.get("company")
        if not title and not company:
            continue
        entry = title or ""
        if company:
            entry = f"{entry} at {company}" if entry else company
        description = exp.get("description")
        if description:
            entry = f"{entry} — {description}"
        lines.append(entry)

    for edu in data.get("education") or []:
        parts = [p for p in (edu.get("degree"), edu.get("field_of_study"), edu.get("school")) if p]
        if parts:
            lines.append("Education: " + ", ".join(parts))

    for proj in data.get("projects") or []:
        name = proj.get("name") or proj.get("title")
        if not name:
            continue
        description = proj.get("description")
        entry = f"{name} — {description}" if description else name
        lines.append(f"Project: {entry}")

    for cert in data.get("certifications") or []:
        name = cert.get("name")
        if name:
            lines.append(f"Certification: {name}")

    location_bits = [loc.get("display") for loc in (data.get("work_location_preferences") or []) if loc.get("display")]
    if location_bits:
        lines.append("Preferred locations: " + ", ".join(location_bits))

    return "\n".join(lines)


def render_job(job_row: Dict) -> str:
    """Deterministic text from a candidate-shard row."""
    lines = []

    title = job_row.get("title")
    if _present(title):
        lines.append(title)

    company = job_row.get("company")
    if _present(company):
        lines.append(f"Company: {company}")

    location_parts = [p for p in (job_row.get("city"), job_row.get("state")) if _present(p)]
    location = ", ".join(location_parts) if location_parts else None
    if job_row.get("is_remote") is True:
        location = f"{location} (Remote)" if location else "Remote"
    if location:
        lines.append(f"Location: {location}")

    exp_years = job_row.get("expected_experience_years")
    if _present(exp_years):
        lines.append(f"Experience required: {exp_years:.0f} years")

    pay_lower = job_row.get("pay_range_lower")
    pay_upper = job_row.get("pay_range_upper")
    ai_salary = job_row.get("ai_salary")
    if _present(pay_lower) and _present(pay_upper):
        lines.append(f"Pay range: ${pay_lower:,.0f} - ${pay_upper:,.0f}")
    elif _present(ai_salary):
        lines.append(f"Estimated salary: ${ai_salary:,.0f}")

    skills = job_row.get("skills")
    if _present(skills):
        skills_list = json.loads(skills) if isinstance(skills, str) else skills
        # Entries are usually plain strings, but some shards store
        # {"name": ..., "popularity": ..., "id": ...} objects instead - handle both.
        skill_names = [s["name"] if isinstance(s, dict) else s for s in skills_list]
        skill_names = [s for s in skill_names if _present(s)]
        if skill_names:
            lines.append("Skills: " + ", ".join(skill_names))

    summary = job_row.get("ai_summary")
    if _present(summary):
        lines.append(summary)

    desc = job_row.get("desc")
    if _present(desc):
        lines.append(desc)

    return "\n".join(lines)


def _shard_filename(shard_index: int) -> str:
    return f"shard_{shard_index:04d}.parquet"


def sample_profile_ids_from_shards(n_profiles: int, seed: int, max_shards: int = NUM_SHARDS) -> Tuple[List[str], List[int]]:
    """Read candidate shards in order, accumulating unique profile_ids until
    n_profiles is reached, then truncate to exactly n_profiles via a seeded
    sample. Shards were checked to be cohort-representative (not grouped), so
    reading in order gives a representative sample without scanning all 125.

    Returns (sampled_profile_ids, shard_indices_read) - the caller reuses
    shard_indices_read to avoid re-downloading the same shards.
    """
    import pyarrow.parquet as pq

    seen_ids: List[str] = []
    seen_set = set()
    shard_indices_read: List[int] = []

    for shard_index in range(max_shards):
        path = hf_hub_download(repo_id=REPO_ID, filename=_shard_filename(shard_index), repo_type=REPO_TYPE)
        shard_indices_read.append(shard_index)
        ids = pq.read_table(path, columns=["profile_id"]).column("profile_id").unique().to_pylist()
        for pid in ids:
            if pid not in seen_set:
                seen_set.add(pid)
                seen_ids.append(pid)
        if len(seen_ids) >= n_profiles:
            break

    if len(seen_ids) < n_profiles:
        raise ValueError(
            f"Only found {len(seen_ids)} unique profiles across {len(shard_indices_read)} shards "
            f"(all {max_shards} shards read) - requested {n_profiles}."
        )

    rng = random.Random(seed)
    sampled = sorted(seen_ids)  # sort first so the seeded sample is deterministic regardless of shard read order
    rng.shuffle(sampled)
    return sampled[:n_profiles], shard_indices_read


def split_profile_ids(
    profile_ids: List[str], seed: int, train_frac: float = 0.8, val_frac: float = 0.1
) -> Tuple[List[str], List[str], List[str]]:
    """Shuffle profile_ids with a fixed seed and split by count into
    (train, val, test). Asserts pairwise-empty intersections before returning."""
    ids = list(profile_ids)
    random.Random(seed).shuffle(ids)

    n = len(ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]

    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)
    assert not (train_set & val_set), "train/val profile_id overlap"
    assert not (train_set & test_set), "train/test profile_id overlap"
    assert not (val_set & test_set), "val/test profile_id overlap"
    assert len(train_set) + len(val_set) + len(test_set) == n, "split sizes don't sum to total"

    return train_ids, val_ids, test_ids


def build_triplets_from_grades(
    train_df,
    seed: int,
    neg_per_positive: int = 4,
    positive_threshold: float = POSITIVE_GRADE_THRESHOLD,
    negative_gap: float = HARD_NEGATIVE_GAP,
) -> List[Dict]:
    """(anchor, positive, negative) triplets from a profile's own graded
    candidates - no separate hard-negative "mining" step needed, since the
    candidates are already a pre-retrieved, plausible pool (that's what makes
    them "candidates" in this dataset).

    Per profile: positive = the highest-graded candidate, if it clears
    positive_threshold (tie-break: highest retrieval_score). Hard negatives =
    candidates at least negative_gap points below that positive grade. Up to
    neg_per_positive of those are sampled per profile. Profiles with no
    qualifying positive, or no qualifying hard negative, are skipped.

    Each returned dict also carries positive_grade/negative_grade (the raw
    0-100 grades behind the pair) alongside anchor/positive/negative - unused
    by the fixed-margin baseline, but lets graded-relevance-aware losses
    (e.g. adaptive-margin) compute the grade gap without re-deriving it.
    """
    rng = random.Random(seed)
    triplets = []

    for _profile_id, group in train_df.groupby("profile_id", sort=True):
        max_grade = group["grade"].max()
        if max_grade < positive_threshold:
            continue

        profile_text = group["profile_text"].iloc[0]
        positive_row = group[group["grade"] == max_grade].sort_values("retrieval_score", ascending=False).iloc[0]
        positive_text = positive_row["job_text"]

        hard_negatives = group.loc[group["grade"] <= max_grade - negative_gap, ["job_text", "grade"]].to_dict("records")
        if not hard_negatives:
            continue

        sample_size = min(neg_per_positive, len(hard_negatives))
        for neg in rng.sample(hard_negatives, sample_size):
            triplets.append({
                "anchor": profile_text,
                "positive": positive_text,
                "negative": neg["job_text"],
                "positive_grade": float(max_grade),
                "negative_grade": float(neg["grade"]),
            })

    return triplets


def evaluate_profiles(model_or_path, profiles_df, k: int = 10) -> Dict[str, float]:
    """Rank each profile's own candidate pool (not a shared corpus - every
    profile here has its own private ~25-job candidate list) and average
    Recall@10, Recall@50, MRR, and graded NDCG@10 across profiles.

    Note: with ~25 candidates per profile, "top 50" always contains the whole
    candidate pool, so a profile with any relevant (grade>=threshold)
    candidate always scores exactly 1.0 on recall@50 - but recall_at_k returns
    0.0 for a profile with zero relevant candidates (a real, common case: see
    the dataset_statistics.json "no Fair+ candidate" rate). So Recall@50 here
    isn't measuring ranking quality (Recall@10 and NDCG@10 already capture
    that within a pool this small) - it converges to roughly the fraction of
    profiles that have any relevant candidate at all, i.e. candidate-pool
    coverage, not ranking skill. Still reported for consistency with every
    other script in this project, just worth reading correctly.

    model_or_path: a model name/checkpoint path, or an in-memory
    SentenceTransformer (the latter avoids a disk round-trip for periodic
    in-training validation checks).
    """
    model = model_or_path if isinstance(model_or_path, SentenceTransformer) else SentenceTransformer(model_or_path)

    profile_ids = profiles_df["profile_id"].unique().tolist()
    profile_text_by_id = profiles_df.drop_duplicates("profile_id").set_index("profile_id")["profile_text"]
    profile_embeddings = model.encode(
        [profile_text_by_id[pid] for pid in profile_ids],
        convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False,
    )
    profile_idx = {pid: i for i, pid in enumerate(profile_ids)}

    job_ids = profiles_df["job_id"].unique().tolist()
    job_text_by_id = profiles_df.drop_duplicates("job_id").set_index("job_id")["job_text"]
    job_embeddings = model.encode(
        [job_text_by_id[jid] for jid in job_ids],
        convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False,
    )
    job_idx = {jid: i for i, jid in enumerate(job_ids)}

    recalls_10, recalls_50, rrs, ndcgs = [], [], [], []

    for profile_id, group in profiles_df.groupby("profile_id", sort=True):
        profile_emb = profile_embeddings[profile_idx[profile_id]]
        candidate_job_ids = group["job_id"].tolist()
        candidate_grades = group["grade"].tolist()
        candidate_embs = job_embeddings[[job_idx[jid] for jid in candidate_job_ids]]

        # Embeddings are already normalized, so cosine similarity = dot product.
        scores = candidate_embs @ profile_emb
        order = scores.argsort(descending=True).tolist()

        ranked_job_ids = [candidate_job_ids[i] for i in order]
        ranked_grades = [candidate_grades[i] for i in order]
        relevant_ids = [jid for jid, g in zip(candidate_job_ids, candidate_grades) if g >= POSITIVE_GRADE_THRESHOLD]

        recalls_10.append(recall_at_k(ranked_job_ids, relevant_ids, k=10))
        recalls_50.append(recall_at_k(ranked_job_ids, relevant_ids, k=50))
        rrs.append(reciprocal_rank(ranked_job_ids, relevant_ids))
        ndcgs.append(graded_ndcg_at_k(ranked_grades, k=k))

    return {
        "recall@10": statistics.mean(recalls_10),
        "recall@50": statistics.mean(recalls_50),
        "mrr": statistics.mean(rrs),
        f"ndcg@{k}": statistics.mean(ndcgs),
    }
