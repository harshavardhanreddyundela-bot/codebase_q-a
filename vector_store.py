"""
Vector Store & Embedding Module.

Manages vector database operations for code chunks.
Supports ChromaDB with persistent storage and Google Gemini embeddings (text-embedding-004).
Includes a local Scikit-Learn TF-IDF vector store fallback if ChromaDB is not yet installed.
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from chunker import CodeChunk

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# Check if chromadb is installed
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

# Check if google-genai is installed
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


class GeminiEmbedder:
    """Generates embeddings using Google Gemini API with rate limiting and quota protection."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-embedding-001"):
        raw_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.api_key = raw_key.strip() if raw_key else None
        self.model = model
        self.client = None
        if self.api_key and HAS_GENAI:
            self.client = genai.Client(api_key=self.api_key)

    def is_available(self) -> bool:
        return bool(self.client is not None)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Batch generate embeddings with rate pacing and exponential backoff for Free Tier quota."""
        if not self.is_available():
            raise RuntimeError("Gemini Client not initialized. Please set GEMINI_API_KEY.")

        embeddings = []
        # Keep batches small (15 chunks) to prevent exceeding the 30k tokens/min rate limit
        batch_size = 15
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for b_idx in range(total_batches):
            start_i = b_idx * batch_size
            end_i = min(start_i + batch_size, len(texts))
            batch = texts[start_i:end_i]

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.client.models.embed_content(
                        model=self.model,
                        contents=batch,
                    )
                    for emb in response.embeddings:
                        embeddings.append(emb.values)
                    # Pace requests to stay within free tier limits (30,000 tokens/min)
                    if b_idx < total_batches - 1:
                        time.sleep(1.2)
                    break
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        if attempt < max_retries - 1:
                            wait_seconds = 15 * (attempt + 1)
                            print(f"[Notice] Rate limit encountered. Waiting {wait_seconds}s before retrying batch {b_idx + 1}...")
                            time.sleep(wait_seconds)
                        else:
                            raise e
                    else:
                        raise e

        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single search query."""
        if not self.is_available():
            raise RuntimeError("Gemini Client not initialized. Please set GEMINI_API_KEY.")

        response = self.client.models.embed_content(
            model=self.model,
            contents=[query],
        )
        return response.embeddings[0].values


class FallbackLocalVectorStore:
    """
    Lightweight vector store fallback using Scikit-Learn TF-IDF cosine similarity.
    Ensures seamless offline development and testing.
    """

    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        self.file_path = os.path.join(persist_dir, "local_vector_store.json")
        self.chunks_data: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self.chunks_data = json.load(f)
            except Exception:
                self.chunks_data = []

    def _save(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks_data, f, indent=2)

    def add_chunks(self, chunks: List[CodeChunk]) -> int:
        for c in chunks:
            self.chunks_data.append({
                "id": c.chunk_id,
                "document": c.content,
                "metadata": c.to_metadata(),
            })
        self._save()
        return len(chunks)

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.chunks_data:
            return []

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        docs = [item["document"] for item in self.chunks_data]
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[a-zA-Z0-9_]{2,}\b")
        tfidf_matrix = vectorizer.fit_transform(docs)
        query_vec = vectorizer.transform([query_text])

        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            results.append({
                "id": self.chunks_data[idx]["id"],
                "document": self.chunks_data[idx]["document"],
                "metadata": self.chunks_data[idx]["metadata"],
                "score": score,
            })
        return results

    def get_all(self) -> List[Dict[str, Any]]:
        return self.chunks_data

    def clear(self):
        self.chunks_data = []
        if os.path.exists(self.file_path):
            os.remove(self.file_path)


class CodebaseVectorStore:
    """
    Main Vector Store interface. Uses ChromaDB with Gemini Embeddings when available,
    or falls back gracefully to Scikit-Learn TF-IDF.
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "codebase_qa",
        api_key: Optional[str] = None,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedder = GeminiEmbedder(api_key=api_key)
        self.use_chroma = HAS_CHROMADB
        self.client = None
        self.collection = None
        self.local_store = None

        if self.use_chroma:
            os.makedirs(self.persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        else:
            self.local_store = FallbackLocalVectorStore(persist_dir=self.persist_dir)

    def add_chunks(self, chunks: List[CodeChunk]) -> int:
        """Add CodeChunk items to the vector store with embeddings and metadata."""
        if not chunks:
            return 0

        if self.use_chroma:
            ids = [c.chunk_id for c in chunks]
            documents = [c.content for c in chunks]
            metadatas = [c.to_metadata() for c in chunks]

            # Generate embeddings with Gemini if available, else let Chroma use default or raw text
            if self.embedder.is_available():
                try:
                    embeddings = self.embedder.embed_texts(documents)
                    self.collection.upsert(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas,
                        embeddings=embeddings,
                    )
                except Exception as e:
                    # If Gemini embedding quota is hit, seamlessly fallback to Chroma's local ONNX embeddings
                    print(f"[Notice] Gemini embedding rate limit reached ({e}). Using local Chroma embeddings.")
                    self.collection.upsert(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas,
                    )
            else:
                # Upsert without pre-computed embeddings (uses Chroma's internal embedder)
                self.collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                )
            return len(chunks)
        else:
            return self.local_store.add_chunks(chunks)

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the vector database for semantically similar code chunks."""
        if self.use_chroma:
            if self.collection.count() == 0:
                return []

            chroma_results = None
            if self.embedder.is_available():
                try:
                    query_emb = self.embedder.embed_query(query_text)
                    chroma_results = self.collection.query(
                        query_embeddings=[query_emb],
                        n_results=min(top_k, self.collection.count()),
                    )
                except Exception:
                    chroma_results = None

            if not chroma_results:
                chroma_results = self.collection.query(
                    query_texts=[query_text],
                    n_results=min(top_k, self.collection.count()),
                )

            # Format normalized output
            results = []
            if chroma_results and "ids" in chroma_results and chroma_results["ids"]:
                ids = chroma_results["ids"][0]
                docs = chroma_results["documents"][0]
                metas = chroma_results["metadatas"][0]
                distances = chroma_results.get("distances", [[0.0] * len(ids)])[0]

                for i in range(len(ids)):
                    # Cosine distance to similarity score
                    sim_score = 1.0 - distances[i] if distances else 1.0
                    results.append({
                        "id": ids[i],
                        "document": docs[i],
                        "metadata": metas[i],
                        "score": sim_score,
                    })
            return results
        else:
            return self.local_store.query(query_text, top_k=top_k)

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Retrieve all indexed chunks and metadata for keyword search & indexing inspection."""
        if self.use_chroma:
            if self.collection.count() == 0:
                return []
            res = self.collection.get()
            all_chunks = []
            for i in range(len(res["ids"])):
                all_chunks.append({
                    "id": res["ids"][i],
                    "document": res["documents"][i],
                    "metadata": res["metadatas"][i],
                })
            return all_chunks
        else:
            return self.local_store.get_all()

    def clear(self):
        """Wipe collection."""
        if self.use_chroma:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        else:
            self.local_store.clear()
