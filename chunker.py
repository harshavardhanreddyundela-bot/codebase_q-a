"""
Code-Aware Chunker Module.

Performs syntax-aware and AST-based chunking on source code repositories,
preserving function, class, and method boundaries while attaching rich metadata
(file path, line numbers, symbol names, language, chunk ID).
"""

import ast
import os
import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional


def resolve_repository_path(repo_path_or_url: str) -> str:
    """
    Resolves a local folder path or automatically clones a GitHub URL
    into a local ./cloned_repos/ directory.
    """
    path_str = repo_path_or_url.strip().strip("'").strip('"')
    if path_str.startswith(("http://", "https://", "git@")):
        repo_name = path_str.rstrip("/").split("/")[-1].replace(".git", "") or "repo"
        cloned_dir = os.path.abspath(os.path.join("./cloned_repos", repo_name))

        if os.path.exists(cloned_dir):
            try:
                subprocess.run(["git", "-C", cloned_dir, "pull"], capture_output=True, text=True, timeout=30)
            except Exception:
                pass
            return cloned_dir

        os.makedirs("./cloned_repos", exist_ok=True)
        res = subprocess.run(
            ["git", "clone", "--depth", "1", path_str, cloned_dir],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if res.returncode != 0:
            raise RuntimeError(f"Failed to clone repository: {res.stderr or res.stdout}")
        return cloned_dir

    if not os.path.exists(path_str):
        raise FileNotFoundError(f"Directory not found: '{path_str}'")

    return os.path.abspath(path_str)


@dataclass
class CodeChunk:
    chunk_id: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    content: str
    symbol_name: str = ""
    symbol_type: str = "block"  # 'function', 'class', 'method', 'module', 'block'
    repo_name: str = ""

    def to_metadata(self) -> Dict[str, Any]:
        """Convert chunk metadata to a flat dictionary suitable for vector databases."""
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "language": self.language,
            "start_line": int(self.start_line),
            "end_line": int(self.end_line),
            "symbol_name": self.symbol_name or "",
            "symbol_type": self.symbol_type or "block",
            "repo_name": self.repo_name or "",
        }

    def format_for_llm(self) -> str:
        """Format chunk with header and line numbers for grounded LLM context."""
        header = f"--- File: {self.file_path} (Lines {self.start_line}-{self.end_line})"
        if self.symbol_name:
            header += f" [{self.symbol_type}: {self.symbol_name}]"
        header += " ---"

        # Prefix each line with its verified line number
        lines = self.content.splitlines()
        annotated_lines = []
        for offset, line in enumerate(lines):
            line_no = self.start_line + offset
            annotated_lines.append(f"{line_no:4d} | {line}")

        return f"{header}\n" + "\n".join(annotated_lines)


class CodeAwareChunker:
    """Chunker that respects code semantics and programming language constructs."""

    SUPPORTED_EXTENSIONS = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".go": "go",
        ".rs": "rust",
        ".php": "php",
        ".rb": "ruby",
        ".html": "html",
        ".css": "css",
        ".sql": "sql",
        ".sh": "bash",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
    }

    IGNORE_DIRS = {
        ".git",
        ".svn",
        ".hg",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        ".idea",
        ".vscode",
        "dist",
        "build",
        ".chroma",
        "chroma_db",
        ".next",
        ".idx",
        "coverage",
        ".turbo",
    }

    IGNORE_FILES = {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "composer.lock",
        "cargo.lock",
        "poetry.lock",
        "gemfile.lock",
        "mix.lock",
        ".ds_store",
    }

    MAX_FILE_BYTES = 100 * 1024  # 100 KB max for a single source file

    def __init__(self, max_chunk_lines: int = 80, overlap_lines: int = 15):
        self.max_chunk_lines = max_chunk_lines
        self.overlap_lines = overlap_lines

    def detect_language(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        return self.SUPPORTED_EXTENSIONS.get(ext, "text")

    def chunk_python(self, code: str, rel_path: str, repo_name: str) -> List[CodeChunk]:
        """Parse Python source using the built-in AST to extract exact functions and classes."""
        chunks: List[CodeChunk] = []
        lines = code.splitlines()
        total_lines = len(lines)

        if total_lines == 0:
            return chunks

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Fall back to sliding window if file has invalid syntax or is Python 2
            return self.chunk_generic(code, rel_path, "python", repo_name)

        covered_ranges = []

        # 1. Module docstring and top imports (if present)
        first_definition_line = total_lines
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                first_definition_line = min(first_definition_line, node.lineno)

        if first_definition_line > 1:
            header_end = min(first_definition_line - 1, total_lines)
            header_content = "\n".join(lines[0:header_end]).strip()
            if header_content:
                chunk_id = f"{rel_path}#L1-L{header_end}"
                chunks.append(
                    CodeChunk(
                        chunk_id=chunk_id,
                        file_path=rel_path,
                        language="python",
                        start_line=1,
                        end_line=header_end,
                        content=header_content,
                        symbol_name="module_header",
                        symbol_type="module",
                        repo_name=repo_name,
                    )
                )
                covered_ranges.append((1, header_end))

        # 2. Extract Functions and Classes with exact line bounds
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = node.lineno
                end = getattr(node, "end_lineno", None) or self._estimate_end_line(lines, start)
                chunk_text = "\n".join(lines[start - 1:end])
                chunks.append(
                    CodeChunk(
                        chunk_id=f"{rel_path}#L{start}-L{end}",
                        file_path=rel_path,
                        language="python",
                        start_line=start,
                        end_line=end,
                        content=chunk_text,
                        symbol_name=node.name,
                        symbol_type="function",
                        repo_name=repo_name,
                    )
                )
                covered_ranges.append((start, end))

            elif isinstance(node, ast.ClassDef):
                start = node.lineno
                end = getattr(node, "end_lineno", None) or self._estimate_end_line(lines, start)
                class_lines_count = end - start + 1

                # Index the class definition
                if class_lines_count <= self.max_chunk_lines:
                    chunk_text = "\n".join(lines[start - 1:end])
                    chunks.append(
                        CodeChunk(
                            chunk_id=f"{rel_path}#L{start}-L{end}",
                            file_path=rel_path,
                            language="python",
                            start_line=start,
                            end_line=end,
                            content=chunk_text,
                            symbol_name=node.name,
                            symbol_type="class",
                            repo_name=repo_name,
                        )
                    )
                else:
                    # Index class header if class is large
                    class_header_end = min(start + 10, end)
                    chunks.append(
                        CodeChunk(
                            chunk_id=f"{rel_path}#L{start}-L{class_header_end}",
                            file_path=rel_path,
                            language="python",
                            start_line=start,
                            end_line=class_header_end,
                            content="\n".join(lines[start - 1:class_header_end]),
                            symbol_name=node.name,
                            symbol_type="class_header",
                            repo_name=repo_name,
                        )
                    )

                # Always extract individual methods so symbol queries (e.g. 'login') map directly to exact line numbers
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        m_start = item.lineno
                        m_end = getattr(item, "end_lineno", None) or self._estimate_end_line(lines, m_start)
                        chunks.append(
                            CodeChunk(
                                chunk_id=f"{rel_path}#L{m_start}-L{m_end}",
                                file_path=rel_path,
                                language="python",
                                start_line=m_start,
                                end_line=m_end,
                                content="\n".join(lines[m_start - 1:m_end]),
                                symbol_name=f"{node.name}.{item.name}",
                                symbol_type="method",
                                repo_name=repo_name,
                            )
                        )
                covered_ranges.append((start, end))

        # 3. If file had no functions/classes or leftover parts, chunk generically
        if not chunks:
            return self.chunk_generic(code, rel_path, "python", repo_name)

        return chunks

    def _estimate_end_line(self, lines: List[str], start_line: int) -> int:
        """Estimate end line using indentation if end_lineno is unavailable."""
        base_indent = len(lines[start_line - 1]) - len(lines[start_line - 1].lstrip())
        for idx in range(start_line, len(lines)):
            line = lines[idx]
            if not line.strip() or line.strip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            if indent <= base_indent:
                return idx
        return len(lines)

    def chunk_generic(
        self, code: str, rel_path: str, language: str, repo_name: str
    ) -> List[CodeChunk]:
        """Chunk non-Python or non-AST code by identifying function/class signatures or sliding window."""
        chunks: List[CodeChunk] = []
        lines = code.splitlines()
        total_lines = len(lines)

        if total_lines == 0:
            return chunks

        # Regex heuristics for detecting symbols in JS, TS, Java, Go, etc.
        symbol_pattern = re.compile(
            r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function|class|interface|type|const|let|var|def|fn|func|public|private|protected)\s+([a-zA-Z0-9_$]+)",
            re.MULTILINE,
        )

        step = max(1, self.max_chunk_lines - self.overlap_lines)
        idx = 0

        while idx < total_lines:
            start_line = idx + 1
            end_line = min(idx + self.max_chunk_lines, total_lines)
            chunk_slice = lines[start_line - 1:end_line]
            chunk_content = "\n".join(chunk_slice)

            # Detect main symbol name in this chunk
            symbol_name = ""
            symbol_type = "block"
            match = symbol_pattern.search(chunk_content)
            if match:
                symbol_name = match.group(1)
                symbol_type = "symbol"

            chunk_id = f"{rel_path}#L{start_line}-L{end_line}"
            chunks.append(
                CodeChunk(
                    chunk_id=chunk_id,
                    file_path=rel_path,
                    language=language,
                    start_line=start_line,
                    end_line=end_line,
                    content=chunk_content,
                    symbol_name=symbol_name,
                    symbol_type=symbol_type,
                    repo_name=repo_name,
                )
            )

            if end_line == total_lines:
                break
            idx += step

        return chunks

    def chunk_file(self, full_path: str, repo_root: str) -> List[CodeChunk]:
        """Read a file and generate CodeChunks with relative paths."""
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            return []

        rel_path = os.path.relpath(full_path, repo_root).replace("\\", "/")
        repo_name = os.path.basename(os.path.abspath(repo_root))
        language = self.detect_language(full_path)

        if language == "python":
            return self.chunk_python(content, rel_path, repo_name)
        else:
            return self.chunk_generic(content, rel_path, language, repo_name)

    def chunk_repository(
        self, repo_path: str, max_files: int = 1000
    ) -> List[CodeChunk]:
        """Traverse a repository directory and return all code chunks."""
        all_chunks: List[CodeChunk] = []
        repo_path = os.path.abspath(repo_path)
        file_count = 0

        for root, dirs, files in os.walk(repo_path):
            # Prune ignored directories in place
            dirs[:] = [d for d in dirs if d not in self.IGNORE_DIRS and not d.startswith(".")]

            for file in sorted(files):
                file_lower = file.lower()
                if file_lower in self.IGNORE_FILES:
                    continue
                if file_lower.endswith((".min.js", ".min.css", ".map", ".bundle.js", ".chunk.js")):
                    continue

                ext = Path(file).suffix.lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    full_path = os.path.join(root, file)
                    try:
                        if os.path.getsize(full_path) > self.MAX_FILE_BYTES:
                            continue
                    except Exception:
                        pass

                    file_chunks = self.chunk_file(full_path, repo_path)
                    all_chunks.extend(file_chunks)
                    file_count += 1
                    if file_count >= max_files:
                        break
            if file_count >= max_files:
                break

        return all_chunks
