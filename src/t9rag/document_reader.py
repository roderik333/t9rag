"""Document reader."""

import os
from pathlib import Path

import pandas as pd
from docx import Document
from odf import teletype, text
from odf.opendocument import load as load_odt
from pypdf import PdfReader

from .vector_store import DocumentDict


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap
    return chunks


def read_file(filepath: str) -> str:
    suffix = Path(filepath).suffix.lower()

    match suffix:
        case ".txt":
            with open(filepath, "r", encoding="utf-8") as file:
                return file.read()
        case ".csv":
            df = pd.read_csv(filepath)
            return df.to_string()
        case ".odt":
            doc = load_odt(filepath)
            return teletype.extractText(doc.getElementsByType(text.P))
        case ".pdf":
            reader = PdfReader(filepath)
            return "".join(page.extract_text() for page in reader.pages)
        case ".docx":
            doc = Document(filepath)
            return "\n".join([para.text for para in doc.paragraphs])
        case _:
            raise ValueError(f"Unsupported file format: {filepath}")


def load_documents(directory: Path, chunk_size: int, chunk_overlap: int) -> list[DocumentDict]:
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        try:
            content = read_file(filepath)
        except ValueError:
            continue  # Skip unsupported file types

        chunks = chunk_text(content, chunk_size, chunk_overlap)
        for i, chunk in enumerate(chunks):
            documents.append(
                {
                    "filename": f"{filename}_chunk_{i}",
                    "content": chunk,
                    "metadata": {
                        "original_file": f"{filepath}",
                        "chunk_index": i,
                    },
                }
            )
    return documents
