"""Data integrity checks for TalentCLEF Task A English development data."""

import sys
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_loader import load_corpus, load_qrels, load_queries  # noqa: E402

SPLIT_DIR = Path(__file__).resolve().parent.parent / "data" / "TaskA" / "development" / "en"


def main() -> None:
    queries = load_queries(SPLIT_DIR)
    corpus = load_corpus(SPLIT_DIR)
    qrels = load_qrels(SPLIT_DIR)

    qrels_query_ids = set(qrels.keys())
    qrels_corpus_ids = {cid for docs in qrels.values() for cid in docs}

    missing_query_ids = sorted(qrels_query_ids - queries.keys())
    missing_corpus_ids = sorted(qrels_corpus_ids - corpus.keys())

    duplicate_pairs = []
    for qid, docs in qrels.items():
        seen = set()
        for cid in docs:
            if cid in seen:
                duplicate_pairs.append((qid, cid))
            seen.add(cid)

    empty_queries = sorted(qid for qid, text in queries.items() if not text.strip())
    empty_corpus = sorted(cid for cid, text in corpus.items() if not text.strip())

    counts = [len(docs) for docs in qrels.values()]

    print("=== Data Integrity Report ===")

    print(f"\nqrels query_ids missing from queries: {len(missing_query_ids)}")
    if missing_query_ids:
        print(f"  {missing_query_ids}")

    print(f"\nqrels document_ids missing from corpus: {len(missing_corpus_ids)}")
    if missing_corpus_ids:
        print(f"  {missing_corpus_ids}")

    print(f"\nDuplicate (query_id, corpus_id) pairs in qrels: {len(duplicate_pairs)}")
    if duplicate_pairs:
        print(f"  {duplicate_pairs}")

    print(f"\nEmpty query texts: {len(empty_queries)}")
    if empty_queries:
        print(f"  {empty_queries}")

    print(f"\nEmpty corpus texts: {len(empty_corpus)}")
    if empty_corpus:
        print(f"  {empty_corpus}")

    print("\nRelevant documents per query:")
    if counts:
        print(f"  min={min(counts)}  max={max(counts)}  mean={mean(counts):.1f}  median={median(counts):.1f}")
        for qid, docs in qrels.items():
            print(f"  {qid}: {len(docs)}")
    else:
        print("  no qrels loaded")


if __name__ == "__main__":
    main()