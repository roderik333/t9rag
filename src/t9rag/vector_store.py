"""Our Vector store."""

from dataclasses import dataclass, field
from typing import TypedDict

from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Embedding, Metadata
from chromadb.utils import batch_utils


class DocumentDict(TypedDict):
    filename: str
    content: str
    embedding: Embedding
    metadata: dict


@dataclass
class VectorStore:
    client: ClientAPI
    collection: Collection = field(init=False)

    def __post_init__(self, persist_directory: str = "./chroma_db"):
        """We post-initialize the VectorStore with a given persist directory."""
        self.collection = self.client.get_or_create_collection("document_collection")

    def add_documents(self, documents: list[DocumentDict]):
        ids = [doc["filename"] for doc in documents]
        embeddings: list[Embedding] = [doc["embedding"] for doc in documents]
        metadata: list[Metadata] = [doc["metadata"] for doc in documents]
        texts = [doc["content"] for doc in documents]
        _collection = batch_utils.create_batches(self.client, ids, embeddings, metadata, texts)
        [self.collection.add(*col) for col in _collection]

    def query(self, query_embedding: Embedding, n_results: int = 3) -> list[dict]:
        results = self.collection.query(query_embeddings=[query_embedding], n_results=n_results)
        if results["documents"] and results["metadatas"]:
            return [
                {"document": doc, "metadata": meta}
                for doc, meta in zip(results["documents"][0], results["metadatas"][0], strict=True)
            ]
        # In case we get nothing..
        return [{"document": "None", "medtadata": "None"}]
