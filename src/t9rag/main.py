"""Where the magic happens."""

import sys
from pathlib import Path
from typing import Any, TypedDict, Unpack

import chromadb
import click
import yaml
from dataclasses import dataclass, field
from .__version__ import __version__
from .document_reader import load_documents
from .embedding_model import EmbeddingModel, get_sentencetransformer
from .ollama_llm import initialize_llm, stream_complete
from .vector_store import DocumentDict, VectorStore


def load_config(config_file: str) -> dict[str, Any]:
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


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


class AskOptions(TypedDict):
    config: str
    model: str
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


@click.command()
@click.option("--directory", default="./documents", help="Directory containing the documents", show_default=True)
@click.option("--model-name", default="NbAiLab/nb-bert-large", help="Name of the HuggingFace model", show_default=True)
@click.option(
    "--db-directory", default="./chroma_db", help="Directory where to store the vector database", show_default=True
)
def read_documents(directory: Path, model_name: str, db_directory: str):
    documents: list[DocumentDict] = load_documents(directory)
    click.echo(f"Loaded {len(documents)} documents.")
    embedding_model = EmbeddingModel(model=get_sentencetransformer(model_name))
    click.secho(f"initializing embedding model: {model_name}", fg="green")

    for doc in documents:
        doc["embedding"] = embedding_model.embed_text(doc["content"])
        click.secho(f"Embeded document {doc['filename']}", fg="green")

    vector_store = VectorStore(client=chromadb.PersistentClient(path=db_directory))
    vector_store.add_documents((documents))
    click.secho(f"Added documents to vector store at {db_directory}", fg="green")


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
@click.pass_context
def ask(ctx: click.Context, **options: Unpack[AskOptions]) -> None:
    if options.get("config"):
        # If config is passed in, the values in config take precedence over CLI options
        # unless an option is explicitly provided
        ctx_dict = {item.name: item.default for item in ctx.command.params}
        config = load_config(options["config"])
        ask_config = config.get("ask", {})
        for key, value in ask_config.items():
            default_value = ctx_dict.get(key)
            if default_value is None or options[key] == default_value:
                options[key] = value
    else:
        config = {"prompts": {}}

    embedding_model = EmbeddingModel(model=get_sentencetransformer(options["model"]))
    vector_store = VectorStore(client=chromadb.PersistentClient(path=options["db_directory"]))

    _llm_options = {
        "model_name": options["llm_model"],
        "timeout": options["llm_timeout"],
        "base_url": options["llm_base_url"],
        "context_window": options["context_window"],
        "verbose": options["verbose"],
        "max_tokens": options["llm_max_tokens"],
        "temperature": options["llm_temperature"],
        "top_p": options["llm_top_p"],
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
                    click.style("Enter your question ", fg="green"),
                    click.style("(or 'exit' to quit)", fg="white"),
                ]
            )
        )
        if query.lower() == "exit":
            break

        query_embedding = embedding_model.embed_text(query)
        results = vector_store.query(query_embedding, n_results=options["n_results"])

        context = "\n\n".join([f"Document: {r['metadata']['filename']}\n{r['document']}" for r in results])
        conversation_history = conversation_memory.get_relevant_history(query, embedding_model)

        prompt_template = get_prompt(options["prompt"], config or {})
        prompt = prompt_template.format(context=context, conversation_history=conversation_history, query=query)

        click.secho("Answer: ", fg="blue", nl=False)

        answer = ""
        for chunk in stream_complete(llm, prompt):
            click.secho(chunk, fg="blue", nl=False)
            sys.stdout.flush()
            answer += chunk
        click.echo()

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
