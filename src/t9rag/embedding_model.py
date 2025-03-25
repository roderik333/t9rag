"""Our embed."""

from dataclasses import dataclass

from chromadb.api.types import Embedding
from sentence_transformers import SentenceTransformer


@dataclass
class EmbeddingModel:
    model: SentenceTransformer  # = field(default=SentenceTransformer("NbAiLab/nb-bert-large"))  # = field(init=False)

    def embed_texts(self, texts: list[str]) -> list[list[float]] | str:
        return self.model.encode(texts).tolist()

    def embed_text(self, text: str) -> Embedding:
        return self.model.encode([text])[0].tolist()
