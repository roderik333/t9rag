"""Where the magic happens."""

from pathlib import Path
from typing import TypedDict, Unpack

import chromadb
import click
from sentence_transformers import SentenceTransformer

from .document_reader import load_documents
from .embedding_model import EmbeddingModel
from .ollama_llm import DEFAULT_CONTEXT_WINDOW, initialize_llm
from .vector_store import DocumentDict, VectorStore


class AskOptions(TypedDict):
    query: str
    model: str
    db_directory: str
    llm_model: str
    llm_base_url: str
    llm_timeout: int
    context_window: int
    verbose: bool


@click.command()
@click.option("--directory", default="./documents", help="Directory containing the documents")
@click.option("--model-name", default="NbAiLab/nb-bert-large", help="Name of the HuggingFace model")
@click.option("--db-directory", default="./chroma_db", help="Directory where to store the vector database")
def read_documents(directory: Path, model_name: str, db_directory: str):
    documents: list[DocumentDict] = load_documents(directory)
    click.echo(f"Loaded {len(documents)} documents.")
    embedding_model = EmbeddingModel(model=SentenceTransformer(model_name))
    click.secho(f"initializing embedding model: {model_name}", fg="green")

    for doc in documents:
        doc["embedding"] = embedding_model.embed_text(doc["content"])
        click.secho(f"Embeded document {doc['filename']}", fg="green")

    vector_store = VectorStore(client=chromadb.PersistentClient(path=db_directory))
    vector_store.add_documents((documents))
    click.secho(f"Added documents to vector store at {db_directory}", fg="green")


@click.command()
@click.option("--query", prompt="Enter your question", help="The question to ask the RAG system")
@click.option("--model", default="NbAiLab/nb-bert-large", help="Name of the HuggingFace model")
@click.option("--db-directory", default="./chroma_db", help="Directory where the vector database is stored")
@click.option("--llm-model", default="llama3.2", help="Name of the Ollama LLM model")
@click.option("--llm-base-url", default="http://localhost:11434", help="Base URL of the Ollama LLM")
@click.option("--llm-timeout", default=600, help="Timeout for the Ollama LLM")
@click.option("--context-window", default=DEFAULT_CONTEXT_WINDOW, help="Context window size for Ollama LLM")
@click.option("--verbose", is_flag=True, help="Enable verbose output")
def ask(**options: Unpack[AskOptions]) -> None:
    embedding_model = EmbeddingModel(model=SentenceTransformer(options["model"]))
    vector_store = VectorStore(client=chromadb.PersistentClient(path=options["db_directory"]))

    _llm_options = {
        "model_name": options["llm_model"],
        "timeout": options["llm_timeout"],
        "base_url": options["llm_base_url"],
        "context_window": options["context_window"],
        "verbose": options["verbose"],
    }

    llm = initialize_llm(
        **_llm_options,  # type: ignore[arg-type]
    )

    if llm is None:
        click.secho("Failed to initialize Ollama LLM", fg="red")
        return

    query_embedding = embedding_model.embed_text(options["query"])
    results = vector_store.query(query_embedding)

    context = "\n\n".join([f"Document: {r['metadata']['filename']}\n{r['document']}" for r in results])
    prompt = f"Based on the following context, please answer the question:\n\nContext:\n{context}\n\nQuestion: {options['query']}\n\nAnswer:"

    response = llm.complete(prompt).text
    click.secho(f"Question: {options['query']}", fg="green")
    click.secho(f"Answer: {response}", fg="blue")


@click.group(help="General commands")
def cli() -> None:
    pass


cli.add_command(read_documents)
cli.add_command(ask)
