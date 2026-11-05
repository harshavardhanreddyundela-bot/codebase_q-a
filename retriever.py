"""
Hybrid Retriever Module.

Combines semantic vector search with exact symbol / keyword matching using
Reciprocal Rank Fusion (RRF) for high accuracy on queries like:
"Where is the login function defined?" or "How does TokenValidator handle expiry?".
"""

import re
from typing import List, Dict, Any, Optional
from vector_store import CodebaseVectorStore
from security import SecurityManager


class HybridRetriever:
    """
    Hybrid retriever fusing semantic vector search with exact symbol/keyword search.
    """

    def __init__(self, vector_store: CodebaseVectorStore):
        self.vector_store = vector_store

    def extract_candidate_symbols(self, query: str) -> List[str]:
        """Extract candidate function/class/variable names from natural language questions."""
        # Find camelCase, snake_case, PascalCase, or alphanumeric identifier words
        tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", query)
        stopwords = {
            "where", "what", "which", "when", "does", "function", "class",
            "method", "defined", "implemented", "handle", "code", "file",
            "explain", "show", "give", "tell", "find", "call", "used", "work"
        }
        candidates = [t for t in tokens if t.lower() not in stopwords]
        return candidates

    def keyword_symbol_search(self, query: str, candidate_symbols: List[str], all_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score chunks based on exact symbol and keyword matches."""
        scored = []
        lower_query = query.lower()

        for chunk in all_chunks:
            meta = chunk.get("metadata", {})
            doc = chunk.get("document", "")
            symbol_name = meta.get("symbol_name", "").lower()
            file_path = meta.get("file_path", "").lower()

            kw_score = 0.0

            # 1. Exact match on chunk symbol_name (e.g. login == login)
            for sym in candidate_symbols:
                sym_lower = sym.lower()
                if sym_lower == symbol_name:
                    kw_score += 10.0
                elif sym_lower in symbol_name:
                    kw_score += 5.0
                elif sym_lower in file_path:
                    kw_score += 3.0
                
                # Check for definition pattern (def login, function login, class Login)
                def_pattern = rf"\b(def|class|function|fn|func|var|let|const)\s+{re.escape(sym_lower)}\b"
                if re.search(def_pattern, doc, re.IGNORECASE):
                    kw_score += 8.0
                elif re.search(rf"\b{re.escape(sym_lower)}\b", doc, re.IGNORECASE):
                    kw_score += 1.5

            if kw_score > 0:
                scored.append({
                    "id": chunk["id"],
                    "document": doc,
                    "metadata": meta,
                    "score": kw_score,
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,
        semantic_weight: float = 1.0,
        keyword_weight: float = 1.2,
    ) -> Dict[str, Any]:
        """
        Execute hybrid retrieval combining semantic similarity and exact keyword/symbol search.
        
        Returns:
            Dict containing:
                - 'chunks': Top retrieved chunks with metadata and scores.
                - 'formatted_context': Grounded context formatted for the LLM.
                - 'verified_files': List of source files present in retrieved chunks.
        """
        # 1. Semantic Search
        semantic_results = self.vector_store.query(query, top_k=top_k * 2)

        # 2. Exact Symbol / Keyword Search
        all_chunks = self.vector_store.get_all_chunks()
        candidate_symbols = self.extract_candidate_symbols(query)
        keyword_results = self.keyword_symbol_search(query, candidate_symbols, all_chunks)

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        # Semantic ranks
        for rank, item in enumerate(semantic_results):
            cid = item["id"]
            chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (semantic_weight / (rrf_k + rank + 1))

        # Keyword ranks
        for rank, item in enumerate(keyword_results):
            cid = item["id"]
            if cid not in chunk_map:
                chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (keyword_weight / (rrf_k + rank + 1))

        # Sort by final fused score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        top_cids = sorted_cids[:top_k]

        final_chunks = []
        formatted_chunk_texts = []
        verified_files = set()

        for cid in top_cids:
            chunk = chunk_map[cid]
            meta = chunk.get("metadata", {})
            file_path = meta.get("file_path", "unknown")
            start_line = meta.get("start_line", 1)
            end_line = meta.get("end_line", 1)
            symbol_name = meta.get("symbol_name", "")
            symbol_type = meta.get("symbol_type", "block")
            content = chunk.get("document", "")

            verified_files.add(file_path)

            # Build line-numbered block
            header = f"--- File: {file_path} (Lines {start_line}-{end_line})"
            if symbol_name:
                header += f" [{symbol_type}: {symbol_name}]"
            header += " ---"

            lines = content.splitlines()
            annotated_lines = []
            for offset, line in enumerate(lines):
                line_no = int(start_line) + offset
                annotated_lines.append(f"{line_no:4d} | {line}")

            formatted_block = f"{header}\n" + "\n".join(annotated_lines)
            formatted_chunk_texts.append(formatted_block)

            final_chunks.append({
                "chunk_id": cid,
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line,
                "symbol_name": symbol_name,
                "symbol_type": symbol_type,
                "content": content,
                "fused_score": rrf_scores[cid],
            })

        # Sanitize and apply untrusted data boundaries
        sanitized_context = SecurityManager.sanitize_untrusted_context(formatted_chunk_texts)

        return {
            "chunks": final_chunks,
            "formatted_context": sanitized_context,
            "verified_files": list(verified_files),
        }
