"""The Ollama class."""

import click
from llama_index.llms.ollama import Ollama

DEFAULT_CONTEXT_WINDOW = 2048


def initialize_llm(
    model_name: str = "llama3.2",
    timeout=600,
    base_url: str = "http://localhost:11434",
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    verbose: bool = False,
) -> Ollama | None:
    try:
        initialized_llm = Ollama(
            model=model_name,
            request_timeout=timeout,
            base_url=base_url,
        )
        if context_window != DEFAULT_CONTEXT_WINDOW:
            initialized_llm.context_window = context_window
            click.secho(
                f"Updated Ollama context window to {context_window}",
                fg="yellow",
            )
        if verbose:
            click.secho(
                f"Initialized Ollama LLM with context window size {initialized_llm.context_window}",
                fg="green",
            )
        return initialized_llm
    except Exception:
        click.secho(
            f"Make sure Ollama is running and the model is downloaded. Run 'ollama pull {model_name}' if needed.",
            fg="red",
        )
        return None
