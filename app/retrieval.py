"""
app/retrieval.py

Reusable retrieval module used by both eval scripts and the main app.
Loads all indexes once at startup, exposes a clean retrieve() interface.

Supports both ChromaDB (local) and Pinecone (cloud) for dense search.

Usage:
    from app.retrieval import Retriever
    r = Retriever()          # loads BM25 + vector store (ChromaDB or Pinecone) + embed model
    chunks = r.retrieve("how does numpy clip work?", method="hybrid", top_k=5)

Environment variables for Pinecone:
    PINECONE_API_KEY  - Required when using Pinecone backend
"""

import logging
import os
import pickle
import re
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def tokenize(text: str) -> list[str]:
    tokens = re.split(
        r"[\s\(\)\[\]\{\}\.,;:\"'=\+\-\*/<>!@#\$%\^&\|\\`~]+",
        text.lower()
    )
    return [t for t in tokens if len(t) > 1]

def get_chunk_id(chunk: dict) -> str:
    if "chunk_id" in chunk:
        return chunk["chunk_id"]
    return (f"{chunk.get('repo','')}::{chunk.get('filepath','')}::"
            f"{chunk.get('function_name','')}::{chunk.get('line_start','')}")

class Retriever:
    """
    Loads all retrieval indexes once and exposes retrieve().
    Designed to be instantiated once at app startup.
    """

    def __init__(self, config_path: str = "config.yaml", strategy: str = "function"):
        self.cfg = load_config(config_path)
        self.ret_cfg = self.cfg["retrieval"]
        self.strategy = strategy
        self._bm25 = None
        self._bm25_chunks = None
        self._collection = None
        self._pinecone_index = None
        self._embed_model = None
        self._loaded = False
        self._vector_backend = self.cfg["vector_store"]["backend"]  # "chroma" or "pinecone"

    def load(self) -> "Retriever":
        """Load all indexes. Call once at startup."""
        log.info("Loading retrieval indexes...")
        self._load_bm25()
        
        if self._vector_backend == "pinecone":
            self._load_pinecone()
        else:
            self._load_chroma()
        
        self._load_embed_model()
        self._loaded = True
        log.info("Retriever ready.")
        return self

    def _load_bm25(self):
        path = Path(self.cfg["bm25"]["index_path"].format(chunking=self.strategy))
        log.info("  Loading BM25 from %s...", path)
        with path.open("rb") as f:
            payload = pickle.load(f)
        self._bm25 = payload["bm25"]
        self._bm25_chunks = payload["chunks"]
        log.info("  BM25 ready (%d docs)", len(self._bm25_chunks))

    def _load_chroma(self):
        import chromadb
        cfg = self.cfg["vector_store"]["chroma"]
        persist_dir = cfg["persist_directory"]
        name = cfg["collection_name"].format(chunking=self.strategy)
        client = chromadb.PersistentClient(path=persist_dir)
        self._collection = client.get_collection(name)
        log.info("  ChromaDB '%s' ready (%d vectors)", name, self._collection.count())

    def _load_pinecone(self):
        """Initialize Pinecone client. Requires PINECONE_API_KEY env var."""
        try:
            from pinecone import Pinecone
        except ImportError:
            raise ImportError("pinecone package not installed. Install with: pip install pinecone-client")
        
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY env variable not set")
        
        cfg = self.cfg["vector_store"]["pinecone"]
        index_name = cfg["index_name"]
        
        pc = Pinecone(api_key=api_key)
        self._pinecone_index = pc.Index(index_name)
        log.info("  Pinecone index '%s' ready", index_name)

    def _load_embed_model(self):
        model_name = self.cfg["embedding"]["model"]
        log.info("  Loading embed model: %s", model_name)
        self._embed_model = SentenceTransformer(model_name)
        log.info("  Embed model ready")

    def _assert_loaded(self):
        if not self._loaded:
            raise RuntimeError("Call Retriever.load() before retrieve()")

    # Retrieval methods 
    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        scores = self._bm25.get_scores(tokenize(query))
        top_idx = scores.argsort()[-top_k:][::-1]
        results = []
        for idx in top_idx:
            c = self._bm25_chunks[idx].copy()
            c["_score"] = float(scores[idx])
            c["_method"] = "bm25"
            results.append(c)
        return results

    def _dense_search(self, query: str, top_k: int) -> list[dict]:
        emb = self._embed_model.encode(
            query, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        ).tolist()
        
        if self._vector_backend == "pinecone":
            return self._dense_search_pinecone(emb, top_k)
        else:
            return self._dense_search_chroma(emb, top_k)
    
    def _dense_search_chroma(self, emb: list[float], top_k: int) -> list[dict]:
        """Query ChromaDB for dense search results."""
        res = self._collection.query(
            query_embeddings=[emb],
            n_results=top_k,
            include=["metadatas", "distances", "documents"],
        )
        chunks = []
        for i, meta in enumerate(res["metadatas"][0]):
            c = dict(meta)
            c["chunk_id"] = res["ids"][0][i]
            c["_score"] = 1.0 - res["distances"][0][i]
            c["_method"] = "dense"
            c["_text"] = res["documents"][0][i]
            chunks.append(c)
        return chunks
    
    def _dense_search_pinecone(self, emb: list[float], top_k: int) -> list[dict]:
        """Query Pinecone for dense search results."""
        # Query Pinecone vector index
        results = self._pinecone_index.query(
            vector=emb,
            top_k=top_k,
            include_metadata=True
        )
        
        chunks = []
        for match in results.get("matches", []):
            meta = match.get("metadata", {})
            c = dict(meta)
            c["chunk_id"] = match["id"]
            c["_score"] = match["score"]  # Pinecone returns similarity score directly
            c["_method"] = "dense"
            # Note: Pinecone metadata should include raw_code if stored during upsert
            chunks.append(c)
        return chunks

    def _rrf_fusion(self, bm25_res: list[dict], dense_res: list[dict]) -> list[dict]:
        k = self.ret_cfg["hybrid"]["rrf_k"]
        scores: dict[str, float] = {}
        chunk_map: dict[str, dict] = {}

        for rank, c in enumerate(bm25_res, 1):
            cid = get_chunk_id(c)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            chunk_map[cid] = c

        for rank, c in enumerate(dense_res, 1):
            cid = get_chunk_id(c)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            chunk_map[cid] = c

        fused = []
        for cid in sorted(scores, key=lambda x: scores[x], reverse=True):
            c = chunk_map[cid].copy()
            c["_score"] = scores[cid]
            c["_method"] = "hybrid_rrf"
            fused.append(c)
        return fused

    def _rerank(self, query: str, chunks: list[dict], top_k: int) -> list[dict]:
        """Cross-encoder reranking. Falls back gracefully if model unavailable."""
        try:
            from sentence_transformers import CrossEncoder
            model_name = self.cfg["retrieval"]["hybrid"]["rerank_model"]
            reranker = CrossEncoder(model_name)
            pairs = [(query, c.get("raw_code", c.get("_text", ""))[:512])
                     for c in chunks]
            scores = reranker.predict(pairs, show_progress_bar=False)
            ranked = sorted(zip(scores, chunks), key=lambda x: -x[0])
            result = []
            for score, c in ranked[:top_k]:
                c = c.copy()
                c["_rerank_score"] = float(score)
                result.append(c)
            return result
        except Exception as e:
            log.warning("Reranking failed (%s) — returning unranked top_k", e)
            return chunks[:top_k]

    # Public interface 
    def retrieve(self, query: str, method: str = "hybrid", top_k: int = 5,
                 rerank: bool = True) -> list[dict]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query:   Natural language question
            method:  "bm25" | "dense" | "hybrid"
            top_k:   Number of final chunks to return
            rerank:  Apply cross-encoder reranking (hybrid only)

        Returns:
            List of chunk dicts with _score, _method fields added.
            Each chunk has: repo, filepath, function_name, class_name,
                           docstring, raw_code (if available), line_start/end
        """
        self._assert_loaded()

        if method == "bm25":
            return self._bm25_search(query, top_k)

        elif method == "dense":
            return self._dense_search(query, top_k)

        elif method == "hybrid":
            cfg = self.ret_cfg["hybrid"]
            bm25_res = self._bm25_search(query, top_k=cfg["bm25_top_k"])
            dense_res = self._dense_search(query, top_k=cfg["dense_top_k"])
            fused = self._rrf_fusion(bm25_res, dense_res)

            if rerank and cfg.get("rerank", True):
                rerank_k = cfg.get("rerank_top_k", top_k)
                # Rerank from a larger pool then take top_k
                candidates = fused[:max(rerank_k * 2, 20)]
                return self._rerank(query, candidates, top_k)
            return fused[:top_k]

        else:
            raise ValueError(f"Unknown method: {method}. Use bm25/dense/hybrid.")

    def format_context(self, chunks: list[dict], max_chars: int = 4000) -> str:
        """
        Format retrieved chunks into a context string for the LLM prompt.
        Format: filepath::functionname (matches system prompt expectations)
        """
        parts = []
        total = 0
        for c in chunks:
            filepath = c.get("filepath", "?")
            fn = c.get("function_name", "?")
            code = c.get("raw_code", c.get("_text", ""))

            # Use filepath::functionname format as specified in system prompt
            header = f"{filepath}::{fn}"
            block = f"{header}\n```python\n{code}\n```"

            if total + len(block) > max_chars:
                # Truncate last chunk rather than drop it
                remaining = max_chars - total - len(header) - 20
                if remaining > 100:
                    block = f"{header}\n```python\n{code[:remaining]}...\n```"
                    parts.append(block)
                break

            parts.append(block)
            total += len(block)

        return "\n\n".join(parts)
