"""Document reader."""

import os
from pathlib import Path

import pandas as pd
from docx import Document
from odf import teletype, text
from odf.opendocument import load as load_odt
from pypdf import PdfReader

from .vector_store import DocumentDict


def load_documents(directory: Path) -> list[DocumentDict]:
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

        documents.append({"filename": filename, "content": content})

    return documents
