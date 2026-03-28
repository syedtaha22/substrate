"""
pipeline/embed_and_upsert.py

Embeds all function chunks and upserts to ChromaDB (local) or Pinecone (cloud).
Backend is controlled by config.yaml (vector_store.backend)

Usage:
    python pipeline/embed_and_upsert.py                  # uses active_profile from config.yaml
    python pipeline/embed_and_upsert.py --profile A3     # override profile
    python pipeline/embed_and_upsert.py --backend pinecone
    python pipeline/embed_and_upsert.py --dry-run        # embed only, don't upsert
"""

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Config 
def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_profile(cfg: dict, profile_name: str | None = None) -> dict:
    name = profile_name or cfg["active_profile"]
    profile = cfg["profiles"][name]
    log.info("Active profile: %s - %s", name, profile["description"])
    return profile, name

# Text builder 
def build_text(chunk: dict, template: str) -> str:
    """
    Build the text string that gets embedded.
    Template from config: "{function_name}\n{docstring}\n{raw_code}"
    """
    return template.format(
        function_name=chunk.get("function_name", ""),
        docstring=chunk.get("docstring", ""),
        raw_code=chunk.get("raw_code", ""),
        class_name=chunk.get("class_name", ""),
        filepath=chunk.get("filepath", ""),
        repo=chunk.get("repo", ""),
    ).strip()

# Data loading 
def load_chunks(cfg: dict, profile_name: str) -> list[dict]:
    """Load all chunks from JSONL files."""
    chunks_dir = Path(cfg["repos"]["chunks_dir"])
    repo_names = cfg["repos"]["names"]

    # For fixed/recursive chunking profiles, use different chunk files
    chunking_strategy = cfg["profiles"][profile_name].get("chunking", "function")

    all_chunks = []
    for repo in repo_names:
        # function-level chunks always come from {repo}.jsonl
        # fixed/recursive chunks come from {repo}_{strategy}.jsonl
        if chunking_strategy == "function":
            jsonl_path = chunks_dir / f"{repo}.jsonl"
        else:
            jsonl_path = chunks_dir / f"{repo}_{chunking_strategy}.jsonl"

        if not jsonl_path.exists():
            if chunking_strategy == "function":
                log.error("Missing: %s - run parse_functions.py first", jsonl_path)
            else:
                log.warning(
                    "Missing: %s - run parse_chunks_fixed.py or parse_chunks_recursive.py",
                    jsonl_path
                )
            continue

        count = 0
        with jsonl_path.open() as f:
            for line in f:
                chunk = json.loads(line)
                all_chunks.append(chunk)
                count += 1
        log.info("  Loaded %6d chunks from %s", count, jsonl_path.name)

    return all_chunks

def stratified_sample(chunks: list[dict], max_n: int) -> list[dict]:
    """
    Sample chunks proportionally by repo to stay within Pinecone free tier.
    Preserves relative distribution across repos.
    """
    if len(chunks) <= max_n:
        return chunks

    log.warning("Sampling %d - %d chunks (stratified by repo)", len(chunks), max_n)

    from collections import defaultdict
    import random

    by_repo: dict[str, list] = defaultdict(list)
    for c in chunks:
        by_repo[c["repo"]].append(c)

    sampled = []
    for repo, repo_chunks in by_repo.items():
        n = max(1, int(max_n * len(repo_chunks) / len(chunks)))
        sampled.extend(random.sample(repo_chunks, min(n, len(repo_chunks))))

    log.info("Sampled %d chunks across %d repos", len(sampled), len(by_repo))
    return sampled


# Embedding 
def embed_chunks(
    chunks: list[dict],
    model_name: str,
    batch_size: int,
    template: str,
) -> tuple[list[str], list[str], list[dict]]:
    """
    Embed all chunks.
    Returns: (ids, texts, embeddings_as_list)
    """
    log.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)

    ids = [c["chunk_id"] for c in chunks]
    texts = [build_text(c, template) for c in chunks]

    log.info("Embedding %d chunks in batches of %d (CPU - this will take a while)...",
             len(chunks), batch_size)
    log.info("Estimated time: ~%.0f minutes", len(chunks) / batch_size / 3)

    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine similarity via dot product
        convert_to_numpy=True,
    )
    duration = time.time() - t0

    log.info(
        "- Embedded %d chunks in %.1fs (%.0f chunks/sec)",
        len(chunks), duration, len(chunks) / duration
    )
    log.info("Embedding matrix shape: %s", embeddings.shape)

    return ids, texts, embeddings


# ChromaDB upsert 
def upsert_chroma(
    chunks: list[dict],
    ids: list[str],
    texts: list[str],
    embeddings: np.ndarray,
    cfg: dict,
    profile_name: str,
) -> None:
    import chromadb

    chroma_cfg = cfg["vector_store"]["chroma"]
    persist_dir = chroma_cfg["persist_directory"]
    # One collection per chunking strategy
    chunking = cfg["profiles"][profile_name].get("chunking", "function") or "none"
    collection_name = chroma_cfg["collection_name"].format(chunking=chunking)

    log.info("Connecting to ChromaDB at: %s", persist_dir)
    client = chromadb.PersistentClient(path=persist_dir)

    # Delete existing collection if it exists (clean slate)
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        log.warning("Collection '%s' already exists - deleting and recreating", collection_name)
        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Upsert in batches (ChromaDB has limits per call)
    BATCH = 500
    total = len(chunks)
    log.info("Upserting %d vectors to collection '%s'...", total, collection_name)

    for i in tqdm(range(0, total, BATCH), desc="Upserting", unit="batch"):
        batch_ids       = ids[i:i+BATCH]
        batch_texts     = texts[i:i+BATCH]
        batch_embeddings = embeddings[i:i+BATCH].tolist()
        batch_chunks    = chunks[i:i+BATCH]

        # Metadata: everything except raw_code (too large for Chroma metadata)
        batch_meta = [
            {
                "repo":          c["repo"],
                "filepath":      c["filepath"],
                "function_name": c["function_name"],
                "class_name":    c["class_name"],
                "is_method":     c["is_method"],
                "line_start":    c["line_start"],
                "line_end":      c["line_end"],
                "line_count":    c["line_count"],
                "docstring":     c["docstring"][:500],   # truncate for metadata
            }
            for c in batch_chunks
        ]

        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_texts,
            metadatas=batch_meta,
        )

    log.info("- Upserted %d vectors to ChromaDB collection '%s'", total, collection_name)
    log.info("  Persist directory: %s", persist_dir)


# Pinecone upsert 
# Untested
def upsert_pinecone(
    chunks: list[dict],
    ids: list[str],
    embeddings: np.ndarray,
    cfg: dict,
) -> None:
    import os
    from pinecone import Pinecone

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not set in environment")

    pc = Pinecone(api_key=api_key)
    index_name = cfg["vector_store"]["pinecone"]["index_name"]
    index = pc.Index(index_name)

    BATCH = 100
    total = len(chunks)
    log.info("Upserting %d vectors to Pinecone index '%s'...", total, index_name)

    for i in tqdm(range(0, total, BATCH), desc="Upserting", unit="batch"):
        batch_ids       = ids[i:i+BATCH]
        batch_embeddings = embeddings[i:i+BATCH].tolist()
        batch_chunks    = chunks[i:i+BATCH]

        vectors = [
            {
                "id": vid,
                "values": emb,
                "metadata": {
                    "repo":          c["repo"],
                    "filepath":      c["filepath"],
                    "function_name": c["function_name"],
                    "class_name":    c["class_name"],
                    "is_method":     c["is_method"],
                    "line_start":    c["line_start"],
                    "line_end":      c["line_end"],
                    "line_count":    c["line_count"],
                    "docstring":     c["docstring"][:500],
                    "raw_code":      c["raw_code"][:1000],
                },
            }
            for vid, emb, c in zip(batch_ids, batch_embeddings, batch_chunks)
        ]
        index.upsert(vectors=vectors)

    log.info("- Upserted %d vectors to Pinecone", total)


# Save embeddings locally (for BM25 + retrieval use) 
def save_embeddings_cache(
    chunks: list[dict],
    ids: list[str],
    texts: list[str],
    embeddings: np.ndarray,
    profile_name: str,
) -> None:
    """Save embeddings + chunk data to disk so we don't re-embed during eval."""
    cache_dir = Path("data/embeddings")
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"embeddings_{profile_name}.pkl"
    payload = {
        "ids": ids,
        "texts": texts,
        "embeddings": embeddings,
        "chunks": chunks,
    }
    with cache_path.open("wb") as f:
        pickle.dump(payload, f)

    size_mb = cache_path.stat().st_size / (1024 * 1024)
    log.info("- Saved embedding cache: %s (%.1f MB)", cache_path, size_mb)


# Main 
def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunks and upsert to vector store")
    parser.add_argument("--profile", type=str, default=None,
                        help="Ablation profile to use (A1–A5 or baseline). Default: from config.yaml")
    parser.add_argument("--backend", type=str, default=None,
                        choices=["chroma", "pinecone"],
                        help="Vector store backend. Default: from config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Embed only - skip upsert. Useful for testing.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Don't save embedding cache to disk.")
    args = parser.parse_args()

    # Load config + env
    from dotenv import load_dotenv
    load_dotenv()

    cfg = load_config()
    profile, profile_name = get_profile(cfg, args.profile)

    if profile.get("chunking") is None and profile_name != "baseline":
        log.error("Profile '%s' has no chunking strategy - nothing to embed.", profile_name)
        return

    backend = args.backend or cfg["vector_store"]["backend"]
    embed_cfg = cfg["embedding"]

    log.info("=" * 65)
    log.info("Substrate - Embed & Upsert")
    log.info("Profile : %s", profile_name)
    log.info("Backend : %s", backend)
    log.info("Model   : %s", embed_cfg["model"])
    log.info("=" * 65)

    # 1. Load chunks
    log.info("\nLoading chunks...")
    chunks = load_chunks(cfg, profile_name)
    if not chunks:
        log.error("No chunks loaded. Aborting.")
        return
    log.info("Total chunks loaded: %d", len(chunks))

    # 2. Sample if needed
    if cfg["sampling"]["enabled"] and backend == "pinecone":
        chunks = stratified_sample(chunks, cfg["sampling"]["max_vectors"])

    # 3. Embed
    ids, texts, embeddings = embed_chunks(
        chunks,
        model_name=embed_cfg["model"],
        batch_size=embed_cfg["batch_size"],
        template=embed_cfg["text_template"],
    )

    # 4. Save cache
    if not args.no_cache:
        save_embeddings_cache(chunks, ids, texts, embeddings, profile_name)

    if args.dry_run:
        log.info("Dry run - skipping upsert.")
        return

    # 5. Upsert
    if backend == "chroma":
        upsert_chroma(chunks, ids, texts, embeddings, cfg, profile_name)
    elif backend == "pinecone":
        upsert_pinecone(chunks, ids, embeddings, cfg)

    log.info("")
    log.info("Done. Next step: python pipeline/build_bm25.py")


if __name__ == "__main__":
    main()
