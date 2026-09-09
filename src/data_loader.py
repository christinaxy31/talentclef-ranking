"""Minimal data loader for TalentCLEF Task A (queries, corpus, qrels)."""

from pathlib import Path
from typing import Dict, List, Union

PathLike = Union[str, Path]


def _load_docs(directory: PathLike) -> Dict[str, str]:
    """Load one document per file, keyed by filename (the original ID)."""
    docs = {}
    for path in sorted(Path(directory).iterdir()):
        if path.is_file():
            docs[path.name] = path.read_text(encoding="utf-8").strip()
    return docs


def load_queries(split_dir: PathLike) -> Dict[str, str]:
    """Load queries/<id> files into {query_id: text}."""
    return _load_docs(Path(split_dir) / "queries")


def load_corpus(split_dir: PathLike) -> Dict[str, str]:
    """Load corpus/<id> files into {corpus_id: text}."""
    return _load_docs(Path(split_dir) / "corpus")


def load_qrels(split_dir: PathLike) -> Dict[str, List[str]]:
    """Load qrels.tsv into {query_id: [relevant_corpus_id, ...]}.

    File format is TREC-style, tab-separated, no header:
    query_id  iteration  corpus_id  relevance
    """
    qrels_path = Path(split_dir) / "qrels.tsv"
    qrels: Dict[str, List[str]] = {}
    with qrels_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            query_id, _iteration, corpus_id, relevance = line.split("\t")
            if relevance != "1":
                continue
            qrels.setdefault(query_id, []).append(corpus_id)
    return qrels
