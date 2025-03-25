"""The Ollama class."""

import click
from llama_index.llms.ollama import Ollama


def initialize_llm(  # noqa [PLR0913]
    model_name: str,
    timeout: int,
    base_url: str,
    context_window: int,
    verbose: bool,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> Ollama | None:
    try:
        initialized_llm = Ollama(
            model=model_name,
            request_timeout=timeout,
            base_url=base_url,
            context_window=context_window,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
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
