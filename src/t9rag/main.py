"""Where the magic happens."""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict, TypeVar, Unpack, cast

import chromadb
import click
import yaml

from .__version__ import __version__
from .document_reader import load_documents
from .embedding_model import EmbeddingModel, get_sentencetransformer
from .ollama_llm import initialize_llm, stream_complete
from .reranker import Document as FilteredDocument
from .reranker import Reranker
from .vector_store import DocumentDict, VectorStore


class ReadDocumentsOptions(TypedDict):
    config: str
    model_name: str
    directory: Path
    db_directory: str
    chunk_size: int
    chunk_overlap: int


class AskOptions(TypedDict):
    config: str
    model: str
    conversation: bool
    db_directory: str
    llm_model: str
    llm_base_url: str
    llm_timeout: int
    context_window: int
    verbose: bool
    llm_max_tokens: int
    llm_temperature: float
    llm_top_p: int
    n_results: int
    prompt: str
    num_gpu: int
    filter_similarities: bool
    similarity_threshold: float
    rerank_documents: bool
    reranker_model: str
    rerank_top_k: int


OptionsType = TypeVar("OptionsType", bound=AskOptions | ReadDocumentsOptions)


def load_config(config_file: str) -> dict[str, Any]:
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def read_config(ctx: click.Context, options: OptionsType, section: str) -> tuple[dict[str, Any], OptionsType]:
    ctx_dict = {item.name: item.default for item in ctx.command.params}
    config = load_config(options["config"])
    ask_config = config.get(section, {})
    for key, value in ask_config.items():
        default_value = ctx_dict.get(key)
        if default_value is None or options[key] == default_value:
            options[key] = value
    return config, options


def get_prompt(prompt_key: str, config: dict[str, Any]) -> str:
    prompts = config.get("prompts", {})
    if prompt_key in prompts:
        return prompts[prompt_key]
    else:
        click.secho(f"Prompt key '{prompt_key}' not found in config. Using default prompt.", fg="yellow")
        return prompts.get(
            "default",
            "Based on the following context, please answer the question:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:",
        )


def filter_results(
    query: str, results: list[dict], embedding_model: EmbeddingModel, threshold: float = 0.5
) -> list[FilteredDocument]:
    query_embedding = embedding_model.embed_text(query)
    filtered_results = []
    for result in results:
        doc_embedding = embedding_model.embed_text(result["document"])
        similarity = embedding_model.calculate_similarity(query_embedding, doc_embedding)
        if similarity >= threshold:
            filtered_results.append(result)
    return filtered_results


@dataclass
class ConversationMemory:
    memory: list[tuple[str, str]] = field(default_factory=list)
    max_turns: int = field(default=5)

    def add_turn(self, question: str, answer: str):
        self.memory.append((question, answer))
        if len(self.memory) > self.max_turns:
            self.memory.pop(0)

    def get_relevant_history(self, current_query: str, embedding_model: EmbeddingModel) -> str:
        if not self.memory:
            return ""
        current_embedding = embedding_model.embed_text(current_query)

        # Calculate similarity scores
        similarities = []
        for q, _ in self.memory:
            q_embedding = embedding_model.embed_text(q)
            similarity = embedding_model.calculate_similarity(current_embedding, q_embedding)
            similarities.append(similarity)

        # Select top 2 most similar interactions
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:2]

        relevant_history = "\n".join([f"Q: {self.memory[i][0]}\nA: {self.memory[i][1]}" for i in top_indices])
        return f"Recent relevant conversation:\n{relevant_history}\n\n"


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to the configuration file.",
)
@click.option("--directory", default="./documents", help="Directory containing the documents", show_default=True)
@click.option("--model-name", default="NbAiLab/nb-bert-large", help="Name of the HuggingFace model", show_default=True)
@click.option(
    "--db-directory", default="./chroma_db", help="Directory where to store the vector database", show_default=True
)
@click.option("--chunk-size", default=1024, help="Size of document chunks", show_default=True)
@click.option("--chunk-overlap", default=20, help="Overlap between chunks", show_default=True)
@click.pass_context
def read_documents(ctx: click.Context, **options: Unpack[ReadDocumentsOptions]):
    if options.get("config"):
        # If config is passed in, the values in config take precedence over CLI options
        # unless an option is explicitly provided
        _, options = read_config(ctx, options, "read-documents")

    documents: list[DocumentDict] = load_documents(
        options["directory"], options["chunk_size"], options["chunk_overlap"]
    )
    click.echo(f"Loaded {len(documents)} documents.")
    embedding_model = EmbeddingModel(model=get_sentencetransformer(options["model_name"]))
    click.secho(f"initializing embedding model: {options['model_name']}", fg="green")

    for doc in documents:
        doc["embedding"] = embedding_model.embed_text(doc["content"])
        click.secho(f"Embeded document {doc['filename']}", fg="green")

    vector_store = VectorStore(client=chromadb.PersistentClient(path=options["db_directory"]))
    vector_store.add_documents((documents))
    click.secho(f"Added document chunks to vector store at {options['db_directory']}", fg="green")


@click.command(help="Ask a question using the RAG system.")
@click.option(
    "--config",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to the configuration file.",
)
@click.option(
    "--prompt",
    default="default",
    help="The key to the prompt value stored in the configuration file",
    show_default=True,
)
@click.option("--conversation", is_flag=True, help="Enable conversation mode", show_default=True)
@click.option("--model", default="NbAiLab/nb-bert-large", help="Name of the HuggingFace model", show_default=True)
@click.option(
    "--db-directory", default="./chroma_db", help="Directory where the vector database is stored", show_default=True
)
@click.option("--llm-model", default="llama3.2", help="Name of the Ollama LLM model", show_default=True)
@click.option("--llm-base-url", default="http://localhost:11434", help="Base URL of the Ollama LLM", show_default=True)
@click.option("--llm-timeout", default=600, help="Timeout for the Ollama LLM", show_default=True)
@click.option("--context-window", default=3090, help="Context window size for Ollama LLM", show_default=True)
@click.option("--verbose", is_flag=True, help="Enable verbose output", show_default=True)
@click.option("--llm-max-tokens", default=1024, help="Maximum number of tokens for the Ollama LLM", show_default=True)
@click.option("--llm-temperature", default=0.3, help="Temperature for the Ollama LLM", show_default=True)
@click.option("--llm-top-p", default=0.5, help="Top-p for the Ollama LLM", show_default=True)
@click.option(
    "--n-results", default=5, help="When querying the vector store, how many of results to return", show_default=True
)
@click.option("--num-gpu", default=0, help="Number of GPUs to use for Ollama LLM", show_default=True)
@click.option("--filter-similarities", is_flag=True, help="Enable filtering similarities", show_default=True)
@click.option(
    "--similarity-threshold", default=0.5, help="Similarity threshold for filtering results", show_default=True
)
@click.option("--rerank-documents", is_flag=True, help="Enable re-ranking documents", show_default=True)
@click.option(
    "--reranker-model",
    default="cross-encoder/ms-marco-MiniLM-L-6-v2",
    help="Name of the reranker model",
    show_default=True,
)
@click.option("--rerank-top-k", default=10, help="Number of top-k results to rerank", show_default=True)
@click.pass_context
def ask(ctx: click.Context, **options: Unpack[AskOptions]) -> None:
    if options.get("config"):
        # If config is passed in, the values in config take precedence over CLI options
        # unless an option is explicitly provided
        # NOTE TO SELF: config contains the complete configuration while options contains only the asked-specific options.
        config, options = read_config(ctx, options, "ask")
    else:
        config = {"prompts": {}}
    embedding_model = EmbeddingModel(model=get_sentencetransformer(options["model"]))
    vector_store = VectorStore(client=chromadb.PersistentClient(path=options["db_directory"]))
    reranker = False
    if options["rerank_documents"]:
        reranker = Reranker(model_name=options["reranker_model"])

    _llm_options = {
        "model_name": options["llm_model"],
        "timeout": options["llm_timeout"],
        "base_url": options["llm_base_url"],
        "context_window": options["context_window"],
        "verbose": options["verbose"],
        "max_tokens": options["llm_max_tokens"],
        "temperature": options["llm_temperature"],
        "top_p": options["llm_top_p"],
        "num_gpu": options["num_gpu"],
    }
    llm = initialize_llm(
        **_llm_options,  # type: ignore[arg-type]
    )

    if llm is None:
        click.secho("Failed to initialize Ollama LLM", fg="red")
        return

    conversation_memory = ConversationMemory()

    while True:
        query = click.prompt(
            "".join(
                [
                    click.style("Enter your question ", fg=config["colors"]["question_text"]),
                    click.style("(or 'exit' to quit)", fg="white"),
                ]
            )
        )
        if query.lower() == "exit":
            break

        query_embedding = embedding_model.embed_text(query)
        results = vector_store.query(query_embedding, n_results=options["n_results"])

        if options["filter_similarities"]:
            results = filter_results(query, results, embedding_model, options["similarity_threshold"])

        if reranker:
            results = reranker.rerank(query, cast("list[FilteredDocument]", results), top_k=options["rerank_top_k"])

        if not results or results[0]["document"] == "None":
            click.secho("No relevant documents found. Try rephrasing your question.", fg="yellow")
            continue

        context = "\n\n".join(
            [
                f"Document: {r['metadata']['original_file']} (Chunk {r['metadata']['chunk_index']})\n{r['document']}"
                for r in results
            ]
        )
        # context = "\n\n".join([f"Document: {r['metadata']['filename']}\n{r['document']}" for r in results])
        conversation_history = conversation_memory.get_relevant_history(query, embedding_model)

        prompt_template = get_prompt(options["prompt"], config or {})
        prompt = prompt_template.format(context=context, conversation_history=conversation_history, query=query)

        click.secho("Answer: ", fg=config["colors"]["answer_text"], nl=False)

        answer = ""
        for chunk in stream_complete(llm, prompt):
            click.secho(chunk, fg=config["colors"]["answer_text"], nl=False)
            sys.stdout.flush()
            answer += chunk
        click.echo()

        # break out unless a conversation is requested
        if not options["conversation"]:
            break

        conversation_memory.add_turn(query, answer)


@click.command(help="Show version")
def version() -> None:
    click.secho(f"t9 RAG CLI v{__version__}", fg="green")


@click.group(help="General commands")
def cli() -> None:
    pass


cli.add_command(version)
cli.add_command(read_documents)
cli.add_command(ask)
