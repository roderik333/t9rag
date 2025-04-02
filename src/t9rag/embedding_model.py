"""Our embed."""

from dataclasses import dataclass

import numpy as np
import torch
from chromadb.api.types import Embedding
from sentence_transformers import SentenceTransformer, models


def get_sentencetransformer(model_name: str) -> SentenceTransformer:
    """Create and retreive a SentenceTransformer.

    NOTE: the sentence transformer startet out as a
    `models.Transformer`.`get_word_embedding_dimension`
    This worked, for the most part. However it broke when using the
    `sentence-transformers/average_word_embeddings_komninos` model.

    I do not know why, but it will at some point have to be investigated.

    Using the `sentence_transformers.SentenceTransformer`.`get_sentence_embedding_dimension`
    seems to be functional with all the models I've tried thus far.

    """
    sentence_embedding_model = SentenceTransformer(model_name)  # type: ignore[arg-type]
    pooling_model = models.Pooling(sentence_embedding_model.get_sentence_embedding_dimension())

    return SentenceTransformer(
        modules=[sentence_embedding_model, pooling_model], device="cuda" if torch.cuda.is_available() else "cpu"
    )


@dataclass
class EmbeddingModel:
    model: SentenceTransformer

    def embed_texts(self, texts: list[str]) -> list[list[float]] | str:
        return self.model.encode(texts).tolist()

    def embed_text(self, text: str) -> Embedding:
        return self.model.encode([text])[0].tolist()

    def calculate_similarity(self, embedding1: Embedding, embedding2: Embedding) -> float:
        # Convert to numpy arrays for easier calculation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Calculate cosine similarity
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
