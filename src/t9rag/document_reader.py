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


def load_documents(directory: Path, chunk_size: int, chunk_overlap: int) -> list[DocumentDict]:
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read()
        elif filename.endswith(".csv"):
            df = pd.read_csv(filepath)
            content = df.to_string()
        elif filename.endswith(".docx"):
            doc = Document(filepath)
            content = "\n".join([para.text for para in doc.paragraphs])
        elif filename.endswith(".odt"):
            doc = load_odt(filepath)
            content = teletype.extractText(doc.getElementsByType(text.P))
        elif filename.endswith(".pdf"):
            reader = PdfReader(filepath)
            content = ""
            for page in reader.pages:
                content += page.extract_text()
        else:
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
        # documents.append({"filename": filename, "content": content})

    return documents
