"""Reranking of documents based on a given query using the cross-encoder model."""

from dataclasses import dataclass, field
from typing import TypedDict

from sentence_transformers import CrossEncoder


class Document(TypedDict):
    document: str
    metadata: dict[str, str]


@dataclass
class Reranker:
    model: CrossEncoder = field(init=False)
    model_name: str

    def __post_init__(self):
        """Post-initialize the Reranker with a given model name. Default is 'cross-encoder/nli-mean-tokens'."""
        self.model = CrossEncoder(self.model_name)

    def rerank(self, query: str, documents: list[Document], top_k: int) -> list[Document]:  # dict[str, str]]:
        pairs = [[query, doc["document"]] for doc in documents]
        scores = self.model.predict(pairs)
        # Create a list of (score, document) tuples
        scored_docs = list(zip(scores, documents, strict=True))
        # Sort by score in descending order and take top_k
        reranked = sorted(scored_docs, key=lambda x: x[0], reverse=True)[:top_k]
        # Return only the documents, discarding the scores
        return [doc for _, doc in reranked]
