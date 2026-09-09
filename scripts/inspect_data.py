"""Inspect TalentCLEF Task A English development data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_loader import load_corpus, load_qrels, load_queries  # noqa: E402

SPLIT_DIR = Path(__file__).resolve().parent.parent / "data" / "TaskA" / "development" / "en"


def preview(text: str, length: int = 100) -> str:
    return text.replace("\n", " ")[:length]


def main() -> None:
    queries = load_queries(SPLIT_DIR)
    corpus = load_corpus(SPLIT_DIR)
    qrels = load_qrels(SPLIT_DIR)

    num_qrels = sum(len(docs) for docs in qrels.values())

    print(f"Number of queries: {len(queries)}")
    print(f"Number of corpus documents: {len(corpus)}")
    print(f"Number of qrels: {num_qrels}")

    print("\nSample queries:")
    for qid in list(queries)[:3]:
        print(f"  [{qid}] {preview(queries[qid])}")

    print("\nSample corpus documents:")
    for cid in list(corpus)[:3]:
        print(f"  [{cid}] {preview(corpus[cid])}")

    sample_qid = next(iter(qrels))
    print(f"\nRelevant documents for query {sample_qid}:")
    for cid in qrels[sample_qid]:
        print(f"  {cid}")


if __name__ == "__main__":
    main()
