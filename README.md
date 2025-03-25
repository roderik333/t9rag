# T9RAG: Text Retrieval Augmented Generation System

> [!NOTE]
> This README.md has been (mostly) written by the LLM it self. That's why it is bragging!!

## Overview

T9RAG is a powerful and flexible Retrieval Augmented Generation (RAG) system designed to process, embed, and query documents using state-of-the-art language models and vector databases. It combines document processing, embedding generation, vector storage, and language model inference to provide accurate and context-aware responses to user queries.

## Features

- Document processing from various file formats
- Embedding generation using Sentence Transformers
- Vector storage and retrieval using ChromaDB
- Language model inference using Ollama
- Customizable context window for LLM
- Command-line interface for easy interaction

## Components

### 1. Document Reader (`document_reader.py`)

Handles loading and processing of documents from a specified directory. Supports various file formats.

### 2. Embedding Model (`embedding_model.py`)

Utilizes Sentence Transformers to generate embeddings for documents and queries.

### 3. Vector Store (`vector_store.py`)

Manages the storage and retrieval of document embeddings using ChromaDB.

### 4. Ollama LLM Interface (`ollama_llm.py`)

Provides an interface to initialize and interact with the Ollama language model.

### 5. Main Application (`main.py`)

Orchestrates the entire RAG process and provides a command-line interface for user interaction.

## Usage

### Reading Documents

```bash
rag read-documents --directory ./documents --model-name NbAiLab/nb-bert-large --db-directory ./chroma_db
```

### Asking questions

```bash
rag ask --llm-model "llama3.2" --llm-base-url "http://87.238.55.118:11434/" --verbose --query "something or other"
```

This command allows you to ask questions based on the processed documents. It retrieves relevant context, generates a response using the LLM, and displays the answer.

### Configuration

- model-name: Specifies the Sentence Transformer model for embedding generation
- db-directory: Sets the directory for the ChromaDB vector store
- llm-model: Chooses the Ollama LLM model
- context-window: Adjusts the context window size for the LLM
- verbose: Enables detailed output for debugging

### Installation

1. Clone the repository
2. Create a virtual environment

```bash

cd t9rag
python -m venv .venv
source .venv/bin/activate
(.venv) python -m pip install .
(.venv) rag --help
```
