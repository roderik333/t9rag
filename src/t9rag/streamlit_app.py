"""Starting streamlit app."""

import os
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import chromadb
import streamlit as st

from .__version__ import __version__
from .embedding_model import EmbeddingModel, get_sentencetransformer
from .main import AskOptions, ConversationMemory, filter_results, get_prompt, load_config, read_documents
from .ollama_llm import initialize_llm, stream_complete
from .reranker import Document as FilteredDocument
from .reranker import Reranker
from .vector_store import VectorStore


def read_ask_options(config_file: Path) -> tuple[dict[str, Any], AskOptions]:
    options: AskOptions = cast("AskOptions", {})
    config = load_config(config_file)
    ask_config = config.get("ask", {})
    for key, value in ask_config.items():
        options[key] = value
    return config, options


@st.cache_resource
def get_llm(options):
    return initialize_llm(**options)


def get_yaml_files(directory="prompts"):
    yaml_files = [f for f in os.listdir(directory) if f.endswith((".yaml"))]
    return yaml_files


def setup_sidebar():
    with st.sidebar:
        st.header("Configuration")
        yaml_files = get_yaml_files()
        selected_config = st.selectbox(
            "Select configuration file",
            yaml_files,
            index=yaml_files.index("default.yaml") if "default.yaml" in yaml_files else 0,
        )
        st.session_state.prompt = {}
        config_file = Path("prompts") / selected_config

        if st.button("Load Config"):
            config, options = read_ask_options(config_file.resolve())
            st.session_state.config = config
            st.session_state.options = options
            st.success(f"Configuration loaded from {config_file}")

        with suppress(KeyError):
            if "config" in st.session_state:
                st.session_state.prompt["name"] = st.selectbox(
                    "Select prompt", st.session_state.config["prompts"].keys(), index=0
                )

        if "config" not in st.session_state:
            st.session_state.config = {}
        if "options" not in st.session_state:
            st.session_state.options = {}

        # Read Documents section
        st.header("Read Documents")
        if st.button("Read Documents"):
            try:
                read_documents(config=config_file)
                st.success("Documents read successfully!")
            except Exception as e:
                st.error(f"Error reading documents: {str(e)}")


def add_to_chat_history(hist: dict, index: int):
    st.session_state.messages[index].append(hist)


def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def stream_response(llm, prompt):
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for chunk in stream_complete(llm, prompt):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    return full_response


def process_user_input(question: str) -> str | None:
    try:
        embedding_model = EmbeddingModel(model=get_sentencetransformer(st.session_state.options["model"]))
        vector_store = VectorStore(client=chromadb.PersistentClient(path=st.session_state.options["db_directory"]))
        reranker = (
            Reranker(model_name=st.session_state.options["reranker_model"])
            if st.session_state.options["rerank_documents"]
            else False
        )

        _llm_options = {
            "model_name": st.session_state.options["llm_model"],
            "timeout": st.session_state.options["llm_timeout"],
            "base_url": st.session_state.options["llm_base_url"],
            "context_window": st.session_state.options["context_window"],
            "verbose": st.session_state.options["verbose"],
            "max_tokens": st.session_state.options["llm_max_tokens"],
            "temperature": st.session_state.options["llm_temperature"],
            "top_p": st.session_state.options["llm_top_p"],
            "num_gpu": st.session_state.options["num_gpu"],
        }
        llm = get_llm(_llm_options)
        if llm is None:
            st.warning("Failed to initialize Ollama LLM", fg="red")
            st.stop()

        conversation_memory = ConversationMemory()

        query_embedding = embedding_model.embed_text(question)
        results = vector_store.query(query_embedding, n_results=st.session_state.options["n_results"])
        if st.session_state.options["filter_similarities"]:
            results = filter_results(
                question, results, embedding_model, st.session_state.options["similarity_threshold"]
            )
            if not results:
                st.warning("No relevant documents found after filtering. You might want to adjust the threshold.")
        if results and reranker:
            results = reranker.rerank(
                question,
                cast("list[FilteredDocument]", results),
                top_k=st.session_state.options["rerank_top_k"],
            )

        if not results or results[0]["document"] == "None":
            st.warning("No relevant documents found. Try rephrasing your question.")
            st.stop()

        context = "\n\n".join(
            [
                f"Document: {r['metadata']['original_file']} (Chunk {r['metadata']['chunk_index']})\n{r['document']}"
                for r in results
            ]
        )
        conversation_history = conversation_memory.get_relevant_history(question, embedding_model)
        prompt_template = get_prompt(st.session_state.prompt.get("name", "default"), st.session_state.config or {})
        prompt = prompt_template.format(context=context, conversation_history=conversation_history, query=question)
        full_response = stream_response(llm, prompt)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        conversation_memory.add_turn(question, full_response)
    except Exception as e:
        st.error(f"Error processing query {str(e)}")


def get_query_results(question, embedding_model, vector_store, reranker):
    query_embedding = embedding_model.embed_text(question)
    results = vector_store.query(query_embedding, n_results=st.session_state.options["n_results"])

    if st.session_state.options["filter_similarities"]:
        results = filter_results(question, results, embedding_model, st.session_state.options["similarity_threshold"])

    if results and reranker:
        results = reranker.rerank(
            question,
            cast("list[FilteredDocument]", results),
            top_k=st.session_state.options["rerank_top_k"],
        )

    return results


def main():
    st.set_page_config(layout="wide", initial_sidebar_state="expanded")

    st.title(f"T9RAG - {__version__}")

    setup_sidebar()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    question = st.chat_input("Enter your question")  # , key="question_input")

    if question and st.session_state.config and st.session_state.options:
        st.session_state.messages.append({"role": "user", "content": question})
        display_chat_history()
        process_user_input(question)
        st.rerun()
    elif not (st.session_state.config and st.session_state.options):
        st.info("Please load a configuration before asking questions.")
    else:
        display_chat_history()


if __name__ == "__main__":
    main()
