"""Our embed."""

from dataclasses import dataclass

import torch
from chromadb.api.types import Embedding
from sentence_transformers import SentenceTransformer, models


def get_sentencetransformer(model_name: str) -> SentenceTransformer:
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    return SentenceTransformer(
        modules=[word_embedding_model, pooling_model], device="cuda" if torch.cuda.is_available() else "cpu"
    )


@dataclass
class EmbeddingModel:
    model: SentenceTransformer

    def embed_texts(self, texts: list[str]) -> list[list[float]] | str:
        return self.model.encode(texts).tolist()

    def embed_text(self, text: str) -> Embedding:
        return self.model.encode([text])[0].tolist()
