"""A simple training script for which I don't know the value of."""

import json
from contextlib import suppress
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import cast

import click
from llama_index.llms.ollama import Ollama

from .document_reader import load_documents

BASEDIR = Path(__file__).parent.parent.parent.resolve()
BREAK_AT = 100


def coerce_json(data: str) -> dict[str, str] | None:
    doc = None
    with suppress(JSONDecodeError):
        doc = json.loads(data)

    if doc is None:
        # try to coerce data to JSON by checking for missing } in string
        with suppress(JSONDecodeError):
            if "}" not in data:
                data = f"{data}}}"
            doc = json.loads(data)

    return doc


def generate_questions(llm: Ollama, chunk: str, prompt: str) -> dict[str, str] | None:
    prompt = prompt.format(chunk=chunk)
    response: str = cast("str", llm.complete(prompt))  # generate(prompt)
    questions = coerce_json(str(response))
    return questions


def process_documents_with_questions(
    directory: Path, chunk_size: int, chunk_overlap: int, llm: Ollama, prompt: str
) -> None:
    documents = load_documents(directory, chunk_size, chunk_overlap)
    processed_documents = []
    num = 0
    for doc in documents:
        questions = generate_questions(llm, doc["content"], prompt)
        if questions is not None:
            for key in questions:
                processed_documents.append({key: questions[key], "chunk": doc["content"]})
                num += 1
        if num == BREAK_AT:
            break

    with open(BASEDIR / "training.json", "w") as fp:
        fp.write(json.dumps(processed_documents))
        click.secho(f"Dumped {num} trainingdata for {num} content chunks")
