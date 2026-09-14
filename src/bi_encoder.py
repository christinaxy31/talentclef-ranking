"""Zero-shot bi-encoder retrieval baseline (no fine-tuning)."""

from typing import Dict, List, Tuple, Optional

from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


class BiEncoder:
    def __init__(
        self,
        corpus: Dict[str, str],
        model_name: str = MODEL_NAME,
        model: Optional[SentenceTransformer] = None,
    ):
        if model is not None:
            self.model = model
        else:
            self.model = SentenceTransformer(model_name)

        self.doc_ids = list(corpus.keys())

        self.doc_embeddings = self.model.encode(
            [corpus[doc_id] for doc_id in self.doc_ids],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def rank(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        top_k = min(top_k, len(self.doc_ids))

        hits = util.semantic_search(
            query_embedding,
            self.doc_embeddings,
            top_k=top_k,
        )[0]

        return [
            (self.doc_ids[hit["corpus_id"]], hit["score"])
            for hit in hits
        ]