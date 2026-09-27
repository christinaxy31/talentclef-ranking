"""Evaluate a Profile-Jobs-Ranked checkpoint (or the zero-shot base model) on
the validation and/or test split, without rerunning training.

Useful when a training script's checkpoint was saved but its
finetune_results*.json wasn't (e.g. Colab disconnected during the final
one-time test evaluation, after model.save() but before the results JSON was
written) - this reuses the exact same evaluate_profiles call the training
scripts make internally, just as a standalone, rerunnable command.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from profile_jobs_ranked import evaluate_profiles  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "profile_jobs_ranked" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model name or local checkpoint directory path")
    parser.add_argument("--split", choices=["validation", "test", "both"], default="test")
    parser.add_argument("--output", default=None, help="Optional path to save results as JSON")
    args = parser.parse_args()

    results = {"model": args.model}

    if args.split in ("validation", "both"):
        val_df = pd.read_parquet(PROCESSED_DIR / "validation.parquet")
        print(f"Evaluating on validation ({val_df['profile_id'].nunique()} profiles)...")
        results["validation"] = evaluate_profiles(args.model, val_df)
        print(f"  {results['validation']}")

    if args.split in ("test", "both"):
        test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
        print(f"Evaluating on test ({test_df['profile_id'].nunique()} profiles)...")
        results["test"] = evaluate_profiles(args.model, test_df)
        print(f"  {results['test']}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
