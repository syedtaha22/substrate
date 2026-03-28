"""
eval/eval_retrieval.py

Evaluates retrieval quality WITHOUT any LLM involvement.
Loads test queries from eval/test_queries.yaml, runs each through
BM25, dense (ChromaDB), and hybrid retrieval, then checks whether
the expected functions appear in the top-K results.

This is the fastest feedback loop - pure retrieval signal, no API calls.

Usage:
    python eval/eval_retrieval.py                      # all queries, all methods
    python eval/eval_retrieval.py --method hybrid      # one method only
    python eval/eval_retrieval.py --tier 2             # cross-repo queries only
    python eval/eval_retrieval.py --query T2-001       # single query
    python eval/eval_retrieval.py --verbose            # show retrieved chunks
"""

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Config 
def load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def load_test_queries(path: str = "eval/test_queries.yaml") -> list[dict]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["queries"]

# Load indexes 
def load_bm25(cfg: dict) -> tuple:
    path = Path(cfg["bm25"]["index_path"].format(chunking="function"))
    log.info("Loading BM25 index from %s...", path)
    with path.open("rb") as f:
        payload = pickle.load(f)
    log.info("  BM25 index ready (%d documents)", len(payload["chunks"]))
    return payload["bm25"], payload["chunks"], payload["texts"]

def load_chroma(cfg: dict):
    import chromadb
    persist_dir = cfg["vector_store"]["chroma"]["persist_directory"]
    collection_name = cfg["vector_store"]["chroma"]["collection_name"].format(
        chunking="function"
    )
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(collection_name)
    log.info("ChromaDB collection '%s' loaded (%d vectors)", 
             collection_name, collection.count())
    return collection

def load_embed_model(cfg: dict) -> SentenceTransformer:
    model_name = cfg["embedding"]["model"]
    log.info("Loading embedding model: %s", model_name)
    return SentenceTransformer(model_name)

# Tokenizer (must match build_bm25.py) 
def tokenize(text: str) -> list[str]:
    import re
    tokens = re.split(r"[\s\(\)\[\]\{\}\.,;:\"'=\+\-\*/<>!@#\$%\^&\|\\`~]+", text.lower())
    return [t for t in tokens if len(t) > 1]

# Retrieval methods 
def retrieve_bm25(
    query: str,
    bm25,
    chunks: list[dict],
    top_k: int = 10,
) -> list[dict]:
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    top_idx = scores.argsort()[-top_k:][::-1]
    results = []
    for idx in top_idx:
        chunk = chunks[idx].copy()
        chunk["_score"] = float(scores[idx])
        chunk["_method"] = "bm25"
        results.append(chunk)
    return results

def retrieve_dense(
    query: str,
    collection,
    model: SentenceTransformer,
    top_k: int = 10,
) -> list[dict]:
    query_embedding = model.encode(
        query, normalize_embeddings=True, convert_to_numpy=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "distances", "documents"],
    )

    chunks = []
    for i, meta in enumerate(results["metadatas"][0]):
        chunk = dict(meta)
        # ChromaDB stores the ID separately - inject it back so rrf_fusion can key on it
        chunk["chunk_id"] = results["ids"][0][i]
        # ChromaDB returns distance (lower = better), convert to similarity
        chunk["_score"] = 1.0 - results["distances"][0][i]
        chunk["_method"] = "dense"
        chunk["_text"] = results["documents"][0][i]
        chunks.append(chunk)
    return chunks

def rrf_fusion(
    bm25_results: list[dict],
    dense_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion - combines two ranked lists.
    score(d) = sum(1 / (k + rank_i(d))) for each list i
    Uses chunk_id as the deduplication key.
    Falls back to function_name::filepath if chunk_id missing.
    """
    def get_id(chunk: dict) -> str:
        if "chunk_id" in chunk:
            return chunk["chunk_id"]
        # Fallback key from metadata fields
        return f"{chunk.get('repo','')}::{chunk.get('filepath','')}::{chunk.get('function_name','')}::{chunk.get('line_start','')}"

    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, chunk in enumerate(bm25_results, 1):
        cid = get_id(chunk)
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        chunk_map[cid] = chunk

    for rank, chunk in enumerate(dense_results, 1):
        cid = get_id(chunk)
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        chunk_map[cid] = chunk

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    fused = []
    for cid in sorted_ids:
        chunk = chunk_map[cid].copy()
        chunk["_score"] = scores[cid]
        chunk["_method"] = "hybrid_rrf"
        fused.append(chunk)
    return fused


def retrieve_hybrid(
    query: str,
    bm25,
    bm25_chunks: list[dict],
    collection,
    model: SentenceTransformer,
    cfg: dict,
    top_k: int = 10,
) -> list[dict]:
    ret_cfg = cfg["retrieval"]
    bm25_results = retrieve_bm25(query, bm25, bm25_chunks, top_k=ret_cfg["hybrid"]["bm25_top_k"])
    dense_results = retrieve_dense(query, collection, model, top_k=ret_cfg["hybrid"]["dense_top_k"])
    fused = rrf_fusion(bm25_results, dense_results, k=ret_cfg["hybrid"]["rrf_k"])
    return fused[:top_k]

# Evaluation logic 
def evaluate_query(
    query_obj: dict,
    results: list[dict],
    top_k: int,
) -> dict:
    """
    Check if must_retrieve functions appear in the top-K results.
    Returns a result dict with hit/miss details.
    """
    must_retrieve = query_obj.get("must_retrieve", [])
    retrieved_functions = [r.get("function_name", "") for r in results[:top_k]]
    retrieved_repos = [r.get("repo", "") for r in results[:top_k]]

    hits = []
    misses = []
    for fn in must_retrieve:
        if fn in retrieved_functions:
            hits.append(fn)
        else:
            misses.append(fn)

    if not must_retrieve:
        # No must_retrieve specified - mark as "unverifiable" (needs human/LLM judge)
        passed = None
        hit_rate = None
    else:
        hit_rate = len(hits) / len(must_retrieve)
        passed = hit_rate >= cfg_threshold

    return {
        "query_id": query_obj["id"],
        "tier": query_obj["tier"],
        "repos": query_obj["repos"],
        "must_retrieve": must_retrieve,
        "hits": hits,
        "misses": misses,
        "hit_rate": hit_rate,
        "passed": passed,
        "retrieved_functions": retrieved_functions[:top_k],
        "retrieved_repos": retrieved_repos[:top_k],
    }


cfg_threshold = 0.5   # module-level so evaluate_query can access it

# Report 
def print_report(
    method: str,
    query_results: list[dict],
    verbose: bool = False,
    retrieval_results: dict | None = None,
) -> dict:
    """Print a formatted report and return summary stats."""
    log.info("")
    log.info("=" * 70)
    log.info("Retrieval Evaluation - Method: %s", method.upper())
    log.info("=" * 70)

    verifiable = [r for r in query_results if r["passed"] is not None]
    unverifiable = [r for r in query_results if r["passed"] is None]
    passed = [r for r in verifiable if r["passed"]]
    failed = [r for r in verifiable if not r["passed"]]

    # Per-tier breakdown
    tiers = sorted(set(r["tier"] for r in query_results))
    for tier in tiers:
        tier_queries = [r for r in verifiable if r["tier"] == tier]
        if not tier_queries:
            continue
        tier_pass = sum(1 for r in tier_queries if r["passed"])
        log.info(
            "  Tier %d: %d/%d passed (%.0f%%)",
            tier, tier_pass, len(tier_queries),
            100 * tier_pass / len(tier_queries) if tier_queries else 0,
        )

    log.info("")
    log.info("  Total verifiable : %d queries", len(verifiable))
    log.info("  Passed           : %d (%.0f%%)",
             len(passed), 100 * len(passed) / len(verifiable) if verifiable else 0)
    log.info("  Failed           : %d", len(failed))
    log.info("  Unverifiable     : %d (no must_retrieve defined - needs LLM judge)", 
             len(unverifiable))

    if failed and verbose:
        log.info("")
        log.info("  Failed queries:")
        for r in failed:
            log.info("    [%s] Tier %d - missed: %s", r["query_id"], r["tier"], r["misses"])

    if verbose and retrieval_results:
        log.info("")
        log.info("  Per-query detail:")
        for qid, chunks in retrieval_results.items():
            log.info("    %s - top 5 retrieved:", qid)
            for i, c in enumerate(chunks[:5], 1):
                log.info(
                    "      %d. [%.4f] %s::%s::%s",
                    i, c.get("_score", 0),
                    c.get("repo", ""), c.get("filepath", "")[:40],
                    c.get("function_name", ""),
                )

    overall_hit_rate = (
        np.mean([r["hit_rate"] for r in verifiable if r["hit_rate"] is not None]) if verifiable else 0.0
    )

    summary = {
        "method": method,
        "total": len(query_results),
        "verifiable": len(verifiable),
        "passed": len(passed),
        "failed": len(failed),
        "unverifiable": len(unverifiable),
        "pass_rate": len(passed) / len(verifiable) if verifiable else 0.0,
        "avg_hit_rate": float(overall_hit_rate),
    }

    log.info("")
    log.info("  Overall hit rate : %.3f", overall_hit_rate)
    log.info("  Pass rate        : %.1f%%", 100 * summary["pass_rate"])
    log.info("=" * 70)

    return summary

def save_results(results: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "retrieval_eval.json"
    with path.open("w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", path)

# Main 
def main() -> None:
    global cfg_threshold

    parser = argparse.ArgumentParser(description="Evaluate retrieval quality (no LLM)")
    parser.add_argument("--method", choices=["bm25", "dense", "hybrid", "all"],
                        default="all")
    parser.add_argument("--tier", type=int, default=None,
                        help="Filter to a specific query tier (1–4)")
    parser.add_argument("--query", type=str, default=None,
                        help="Run a single query by ID (e.g. T2-001)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--verbose", action="store_true",
                        help="Show retrieved chunks per query")
    args = parser.parse_args()

    cfg = load_config()
    cfg_threshold = cfg["evaluation"]["thresholds"]["retrieval_hit_rate"]

    # Load test queries
    queries = load_test_queries(cfg["evaluation"]["test_queries_path"])
    if args.tier:
        queries = [q for q in queries if q["tier"] == args.tier]
    if args.query:
        queries = [q for q in queries if q["id"] == args.query]
    if not queries:
        log.error("No queries matched filters.")
        sys.exit(1)

    log.info("Running retrieval eval on %d queries (top_k=%d)", len(queries), args.top_k)

    # Load indexes
    bm25, bm25_chunks, bm25_texts = load_bm25(cfg)
    collection = load_chroma(cfg)
    model = load_embed_model(cfg)

    methods_to_run = (
        ["bm25", "dense", "hybrid"] if args.method == "all" else [args.method]
    )

    all_summaries = []
    output_data = {}

    for method in methods_to_run:
        log.info("\nRunning method: %s", method)
        query_results = []
        retrieval_details = {}
        t0 = time.time()

        for q in queries:
            if method == "bm25":
                results = retrieve_bm25(q["query"], bm25, bm25_chunks, top_k=args.top_k)
            elif method == "dense":
                results = retrieve_dense(q["query"], collection, model, top_k=args.top_k)
            else:
                results = retrieve_hybrid(
                    q["query"], bm25, bm25_chunks, collection, model, cfg, top_k=args.top_k
                )

            eval_result = evaluate_query(q, results, args.top_k)
            query_results.append(eval_result)
            retrieval_details[q["id"]] = results

        duration = time.time() - t0
        log.info("  Completed in %.1fs (%.2fs/query)", duration, duration / len(queries))

        summary = print_report(
            method, query_results,
            verbose=args.verbose,
            retrieval_results=retrieval_details if args.verbose else None,
        )
        all_summaries.append(summary)
        output_data[method] = {
            "summary": summary,
            "per_query": query_results,
        }

    # Final comparison table
    if len(all_summaries) > 1:
        log.info("")
        log.info("=" * 70)
        log.info("Comparison Summary")
        log.info("=" * 70)
        log.info("  %-10s  %8s  %10s  %10s", "Method", "Passed", "Pass Rate", "Hit Rate")
        log.info("  " + "-" * 45)
        for s in all_summaries:
            log.info(
                "  %-10s  %8d  %9.1f%%  %9.3f",
                s["method"], s["passed"],
                100 * s["pass_rate"], s["avg_hit_rate"],
            )
        log.info("=" * 70)

    save_results(output_data, Path(cfg["evaluation"]["results_dir"]))
    log.info("\nNext step: python eval/eval_baseline.py  (test raw LLM with no RAG)")


if __name__ == "__main__":
    main()