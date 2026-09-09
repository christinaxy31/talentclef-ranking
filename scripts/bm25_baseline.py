"""Smallest possible BM25 baseline: rank the corpus for one job query."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bm25 import BM25  # noqa: E402
from data_loader import load_corpus, load_queries  # noqa: E402

SPLIT_DIR = Path(__file__).resolve().parent.parent / "data" / "TaskA" / "development" / "en"


def preview(text: str, length: int = 100) -> str:
    return text.replace("\n", " ")[:length]


def main() -> None:
    queries = load_queries(SPLIT_DIR)
    corpus = load_corpus(SPLIT_DIR)

    query_id = next(iter(queries))
    query_text = queries[query_id]

    bm25 = BM25(corpus)
    top_10 = bm25.rank(query_text, top_k=10)

    print(f"Query [{query_id}]: {preview(query_text)}")
    print("\nTop 10 resumes by BM25 score:")
    for rank, (doc_id, score) in enumerate(top_10, start=1):
        print(f"{rank:2d}. [{doc_id}] score={score:.4f}  {preview(corpus[doc_id])}")


if __name__ == "__main__":
    main()