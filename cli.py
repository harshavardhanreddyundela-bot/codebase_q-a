"""
CLI Interface for Codebase Q&A Assistant.

Commands:
  python cli.py index <repo_path>   -> Parse and index codebase into vector store
  python cli.py ask "<question>"     -> Ask a grounded question against the indexed code
  python cli.py inspect              -> View all indexed files and symbols
  python cli.py clear                -> Wipe the vector database
"""

import sys
import os
import argparse
from chunker import CodeAwareChunker, resolve_repository_path
from vector_store import CodebaseVectorStore
from rag_engine import CodebaseQARagEngine


def cmd_index(repo_path: str):
    try:
        resolved_path = resolve_repository_path(repo_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"\n[+] Scanning and chunking repository: {resolved_path}")
    chunker = CodeAwareChunker()
    chunks = chunker.chunk_repository(resolved_path)

    if not chunks:
        print("[!] No supported code files found to index.")
        return

    print(f"[+] Found {len(chunks)} code chunks across repository.")
    functions = [c for c in chunks if c.symbol_type in ("function", "method")]
    classes = [c for c in chunks if c.symbol_type == "class"]
    print(f"    - Functions/methods: {len(functions)}")
    print(f"    - Classes: {len(classes)}")

    print("[+] Storing chunks into Vector Database...")
    store = CodebaseVectorStore()
    store.clear()
    count = store.add_chunks(chunks)
    print(f"[OK] Successfully indexed {count} chunks into Vector DB!\n")


def cmd_ask(question: str):
    print(f"\n[?] Question: {question}")
    engine = CodebaseQARagEngine()
    result = engine.generate_grounded_answer(question)

    print("\n" + "=" * 60)
    print(result["answer"])
    print("=" * 60)

    if result["chunks"]:
        print("\n--- Retrieved Grounded Evidence ---")
        for idx, c in enumerate(result["chunks"], 1):
            sym = f" [{c['symbol_type']}: {c['symbol_name']}]" if c.get("symbol_name") else ""
            print(f"[{idx}] {c['file_path']} (Lines {c['start_line']}-{c['end_line']}){sym}")

    if result.get("unverified_files"):
        print(f"\n[!] Warning: Model cited files not verified in indexed chunks: {result['unverified_files']}")


def cmd_inspect():
    store = CodebaseVectorStore()
    chunks = store.get_all_chunks()
    if not chunks:
        print("Vector database is empty. Run `python cli.py index <path>` first.")
        return

    print(f"\n--- Indexed Chunks ({len(chunks)} total) ---")
    files = {}
    for c in chunks:
        meta = c.get("metadata", {})
        fp = meta.get("file_path", "unknown")
        files.setdefault(fp, []).append(meta)

    for fp, metas in files.items():
        print(f"\nFile: {fp} ({len(metas)} chunks)")
        for m in metas:
            sym = f" -> {m.get('symbol_type')}: {m.get('symbol_name')}" if m.get("symbol_name") else ""
            print(f"   L{m.get('start_line')}-L{m.get('end_line')}{sym}")


def cmd_clear():
    store = CodebaseVectorStore()
    store.clear()
    print("[OK] Vector store cleared successfully.")


def main():
    parser = argparse.ArgumentParser(description="Codebase Q&A Assistant CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Index command
    idx_parser = subparsers.add_parser("index", help="Index a repository folder")
    idx_parser.add_argument("path", help="Path to repository")

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question about the indexed repository")
    ask_parser.add_argument("question", help="Question text (e.g. 'Where is the login function defined?')")

    # Inspect command
    subparsers.add_parser("inspect", help="Inspect indexed chunks and symbols")

    # Clear command
    subparsers.add_parser("clear", help="Clear vector database")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args.path)
    elif args.command == "ask":
        cmd_ask(args.question)
    elif args.command == "inspect":
        cmd_inspect()
    elif args.command == "clear":
        cmd_clear()


if __name__ == "__main__":
    main()
