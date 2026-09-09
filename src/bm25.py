"""Minimal BM25 implementation (no external dependencies)."""

import math
import re
from typing import Dict, List, Tuple

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus: Dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        self.doc_ids = list(corpus.keys())
        doc_tokens = {doc_id: tokenize(text) for doc_id, text in corpus.items()}
        self.doc_lens = {doc_id: len(tokens) for doc_id, tokens in doc_tokens.items()}
        self.avgdl = sum(self.doc_lens.values()) / len(self.doc_lens)
        self.n_docs = len(self.doc_ids)

        self.term_freqs: Dict[str, Dict[str, int]] = {}
        self.doc_freqs: Dict[str, int] = {}
        for doc_id, tokens in doc_tokens.items():
            counts: Dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self.term_freqs[doc_id] = counts
            for term in counts:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

    def _idf(self, term: str) -> float:
        n_qi = self.doc_freqs.get(term, 0)
        return math.log((self.n_docs - n_qi + 0.5) / (n_qi + 0.5) + 1)

    def score(self, query: str, doc_id: str) -> float:
        doc_term_freqs = self.term_freqs[doc_id]
        doc_len = self.doc_lens[doc_id]
        total = 0.0
        for term in tokenize(query):
            f = doc_term_freqs.get(term, 0)
            if f == 0:
                continue
            idf = self._idf(term)
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            total += idf * numerator / denominator
        return total

    def rank(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        scores = [(doc_id, self.score(query, doc_id)) for doc_id in self.doc_ids]
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]