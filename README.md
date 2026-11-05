# Codebase Q&A Assistant (RAG)

An AI software engineering assistant that helps developers understand, analyze, and debug code repositories using Retrieval-Augmented Generation (RAG).

All answers are grounded in the retrieved code context with exact file paths, symbols, and verified line numbers.

---

## Features

- **Code-Aware Chunking (`chunker.py`)**: Uses Python's Abstract Syntax Tree (AST) to chunk code along function and class boundaries, capturing exact line ranges and symbol definitions.
- **Rich Metadata Extraction**: Tracks `file_path`, `language`, `start_line`, `end_line`, `symbol_name`, and `symbol_type`.
- **ChromaDB & Vector Store (`vector_store.py`)**: Persistent vector store with Google Gemini embeddings (`text-embedding-004`) and local fallback.
- **Hybrid Retrieval (`retriever.py`)**: Combines vector semantic search with exact keyword/symbol matching using **Reciprocal Rank Fusion (RRF)**.
- **Security & Trust Guardrails (`security.py`)**:
  - Automatically redacts API keys, credentials, and secrets.
  - Treats all repository comments and docstrings as untrusted data (`<untrusted_code_context>`) to neutralize prompt injection.
  - Validates source citations to eliminate hallucinated file paths.
- **Grounded LLM Generation (`rag_engine.py`)**:
  - Employs the strict Codebase Q&A system prompt.
  - Generates responses strictly following:
    - **Answer**
    - **Evidence** (file path, line numbers, symbols)
    - **Explanation**
    - **Suggested fix** (when applicable)
- **Interactive Web UI (`app.py`)** & **CLI (`cli.py`)**.

---

## Project Structure

```text
├── chunker.py         # AST and syntax-aware code chunker
├── security.py        # Secret redaction, untrusted data sandboxing, source verification
├── vector_store.py    # ChromaDB & Gemini embedding integration (with local fallback)
├── retriever.py       # Hybrid semantic + exact symbol search (RRF)
├── rag_engine.py      # Core RAG generator with system prompt & citation enforcement
├── cli.py             # Command-line interface (index, ask, inspect, clear)
├── app.py             # Streamlit web chat UI with collapsible evidence viewer
├── test_rag.py        # Verification test suite
├── requirements.txt   # Dependencies
└── .env.example       # API key configuration template
```

---

## Quick Start

### 1. Configure API Key
Copy `.env.example` to `.env` and insert your Gemini API Key:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
```

### 2. Run the Test Suite
Verify that all components work correctly:
```bash
python test_rag.py
```

### 3. Use the CLI
Index any repository or directory:
```bash
python cli.py index "C:/path/to/your/project"
```

Ask a grounded question:
```bash
python cli.py ask "Where is the login function defined?"
```

Inspect indexed chunks and symbols:
```bash
python cli.py inspect
```

### 4. Launch the Streamlit Web UI
Run the interactive web interface:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`. Enter your repository directory and start chatting!
