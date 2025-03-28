"""The Ollama class."""

import contextlib
from collections.abc import Iterator

import click
from llama_index.llms.ollama import Ollama


def stream_complete(llm: Ollama, prompt: str) -> Iterator[str]:
    response = llm.stream_complete(prompt)
    for chunk in response:
        if chunk.delta:
            yield chunk.delta


def get_gpu_info():
    """Get GPU information if available.

    (this is stackoverflow code.. hope it works)

    """
    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return f"Total GPU memory: {info.total / 1e9:.2f} GB, Used: {info.used / 1e9:.2f} GB, Free: {info.free / 1e9:.2f} GB"
        return None
    except ImportError:
        return "pynvml not installed. Cannot retrieve GPU information."
    except Exception as e:
        return f"Error retrieving GPU information: {str(e)}"
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()


def initialize_llm(  # noqa [PLR0913]
    model_name: str,
    timeout: int,
    base_url: str,
    context_window: int,
    verbose: bool,
    max_tokens: int,
    temperature: float,
    top_p: float,
    num_gpu: int = -1,  # Add this parameter
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
            num_gpu=num_gpu if num_gpu >= 0 else None,  # Add this parameter
        )

        if verbose:
            click.secho(
                f"Initialized Ollama LLM with context window size {initialized_llm.context_window}",
                fg="green",
            )

            # Check GPU usage
            gpu_info = get_gpu_info()
            if gpu_info:
                click.secho(f"GPU in use: {gpu_info}", fg="green")
            else:
                click.secho("No GPU detected or in use", fg="yellow")

        return initialized_llm
    except Exception as e:
        click.secho(
            f"Error initializing Ollama: {str(e)}. Make sure Ollama is running and the model is downloaded. Run 'ollama pull {model_name}' if needed.",
            fg="red",
        )
        return None
