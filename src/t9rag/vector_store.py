"""Our Vector store."""

from dataclasses import dataclass, field
from typing import TypedDict

from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Embedding, Metadata


class DocumentDict(TypedDict):
    filename: str
    content: str
    embedding: Embedding


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
        metadatas: list[Metadata] = [{"filename": doc["filename"]} for doc in documents]
        texts = [doc["content"] for doc in documents]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts,
        )

    def query(self, query_embedding: Embedding, n_results: int = 3) -> list[dict]:
        results = self.collection.query(query_embeddings=[query_embedding], n_results=n_results)
        if results["documents"] and results["metadatas"]:
            return [
                {"document": doc, "metadata": meta}
                for doc, meta in zip(results["documents"][0], results["metadatas"][0], strict=True)
            ]
        # In case we get nothing..
        return [{"document": "None", "medtadata": "None"}]
