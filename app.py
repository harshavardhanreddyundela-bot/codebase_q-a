"""
Streamlit Web UI for Codebase Q&A Assistant.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

from chunker import CodeAwareChunker, resolve_repository_path
from vector_store import CodebaseVectorStore
from rag_engine import CodebaseQARagEngine, RAG_SYSTEM_PROMPT

st.set_page_config(
    page_title="Codebase Q&A Assistant",
    page_icon="💻",
    layout="wide",
)

st.title("💻 Codebase Q&A Assistant (RAG)")
st.caption("Understand, analyze, and debug code repositories using grounded Retrieval-Augmented Generation.")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    api_key_input = st.text_input(
        "Gemini API Key",
        value=env_key,
        type="password",
        help="Your key is automatically loaded from the permanent .env file.",
    )
    # Prefer non-empty input, otherwise fall back to .env key
    active_api_key = (api_key_input.strip() if api_key_input.strip() else env_key).strip()
    if active_api_key:
        os.environ["GEMINI_API_KEY"] = active_api_key
        st.caption("🟢 API Key is configured and ready.")
    else:
        st.warning("⚠️ No API Key found.")

    model_choice = st.selectbox(
        "LLM Model",
        ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash"],
        index=0,
    )
    top_k = st.slider("Retrieved Chunks (Top-K)", min_value=1, max_value=10, value=4)

    st.divider()
    st.subheader("📁 Index Repository")
    repo_input = st.text_input(
        "Repository Directory Path or GitHub URL",
        value=".",
        help="Enter a local folder path (e.g. . or C:/myrepo) or a public GitHub URL (e.g. https://github.com/user/repo)",
    )
    col1, col2 = st.columns(2)
    with col1:
        index_btn = st.button("Index Codebase", use_container_width=True)
    with col2:
        clear_btn = st.button("Clear Vector DB", use_container_width=True)

    if clear_btn:
        store = CodebaseVectorStore()
        store.clear()
        st.session_state["indexed_repo_name"] = None
        st.success("Vector DB cleared!")

    if index_btn:
        try:
            with st.spinner("Resolving repository (cloning if GitHub URL)..."):
                resolved_dir = resolve_repository_path(repo_input)

            with st.spinner(f"Parsing code & building AST chunks from {os.path.basename(resolved_dir)}..."):
                chunker = CodeAwareChunker()
                chunks = chunker.chunk_repository(resolved_dir)
                if not chunks:
                    st.warning("No code files found in the specified repository.")
                else:
                    store = CodebaseVectorStore(api_key=active_api_key or None)
                    store.clear()
                    store.add_chunks(chunks)
                    repo_label = os.path.basename(resolved_dir)
                    st.session_state["indexed_repo_name"] = repo_label
                    st.success(f"Indexed {len(chunks)} chunks from '{repo_label}'!")
                    
                    functions = [c for c in chunks if c.symbol_type in ("function", "method")]
                    classes = [c for c in chunks if c.symbol_type == "class"]
                    st.metric("Total Chunks", len(chunks))
                    st.metric("Functions/Methods", len(functions))
                    st.metric("Classes", len(classes))
        except FileNotFoundError as fnf:
            st.error(f"Directory not found: {fnf}")
        except Exception as e:
            st.error(f"Error indexing repository: {e}")

    if st.session_state.get("indexed_repo_name"):
        st.info(f"Active Index: **{st.session_state['indexed_repo_name']}**")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Quick sample questions
st.markdown("**Quick Prompts:**")
sample_cols = st.columns(4)
sample_prompts = [
    "Where is the login function defined?",
    "Explain the architecture of this project.",
    "Are there any security issues or hardcoded secrets?",
    "How does error handling work in this repository?",
]

selected_prompt = None
for i, col in enumerate(sample_cols):
    if col.button(sample_prompts[i], key=f"sample_{i}", use_container_width=True):
        selected_prompt = sample_prompts[i]

# Display Existing Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "chunks" in msg and msg["chunks"]:
            with st.expander(f"🔍 Grounded Evidence ({len(msg['chunks'])} chunks)"):
                for idx, c in enumerate(msg["chunks"], 1):
                    sym = f" [{c['symbol_type']}: `{c['symbol_name']}`]" if c.get("symbol_name") else ""
                    st.markdown(f"**{idx}. `{c['file_path']}`** (Lines {c['start_line']}–{c['end_line']}){sym}")
                    st.code(c["content"], language=c.get("language", "python"))

# User Input
user_query = st.chat_input("Ask a question about the repository...") or selected_prompt

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving grounded context & analyzing..."):
            engine = CodebaseQARagEngine(
                model_name=model_choice,
                api_key=active_api_key or None,
            )
            result = engine.generate_grounded_answer(user_query, top_k=top_k)

        st.markdown(result["answer"])

        if result["chunks"]:
            with st.expander(f"🔍 Grounded Evidence ({len(result['chunks'])} chunks)"):
                for idx, c in enumerate(result["chunks"], 1):
                    sym = f" [{c['symbol_type']}: `{c['symbol_name']}`]" if c.get("symbol_name") else ""
                    st.markdown(f"**{idx}. `{c['file_path']}`** (Lines {c['start_line']}–{c['end_line']}){sym}")
                    st.code(c["content"], language=c.get("language", "python"))

        if result.get("unverified_files"):
            st.warning(f"⚠️ Unverified file citations detected: {', '.join(result['unverified_files'])}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "chunks": result["chunks"],
        })
