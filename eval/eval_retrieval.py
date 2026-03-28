"""
eval/eval_retrieval.py

Evaluates retrieval quality WITHOUT any LLM involvement.

Scoring approach (v2 - keyword-in-context):
  - Retrieve top-K chunks for each query
  - Concatenate retrieved chunk text into a context string
  - Check what % of context_keywords appear in that context
  - Pass if keyword coverage >= threshold (default 0.5)
  - must_retrieve kept as optional strict secondary check

This measures "does the retrieved context contain the right concepts?"
rather than "did we find these exact function names?" - a much more
meaningful signal for a corpus of 81k chunks.


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
import re
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
    return payload["bm25"], payload["chunks"]

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

def get_id(chunk: dict) -> str:
    return chunk.get("chunk_id") or (
        f"{chunk.get('repo','')}::{chunk.get('filepath','')}::"
        f"{chunk.get('function_name','')}::{chunk.get('line_start','')}"
    )

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
    ret_cfg = cfg["retrieval"]["hybrid"]
    bm25_results = retrieve_bm25(query, bm25, bm25_chunks, top_k=ret_cfg["bm25_top_k"])
    dense_results = retrieve_dense(query, collection, model, top_k=ret_cfg["dense_top_k"])
    return rrf_fusion(bm25_results, dense_results, k=ret_cfg["rrf_k"])[:top_k]

# Evaluation logic
def build_context(chunks: list[dict]) -> str:
    """Concatenate all retrieved chunk text into one searchable string."""
    parts = []
    for c in chunks:
        parts.append(f"{c.get('function_name','')} "
                     f"{c.get('docstring','')} "
                     f"{c.get('raw_code', c.get('_text',''))}")
    return " ".join(parts).lower()

def evaluate_query(
    query_obj: dict,
    chunks: list[dict],
    top_k: int,
    kw_threshold: float = 0.5,
) -> dict:
    """
    Primary metric - keyword-in-context:
        Does the retrieved context contain the query's expected concepts?
        context_keywords field in test_queries.yaml defines what to look for.
        Falls back to keywords field if context_keywords not present.
        Score = found / total. Pass if score >= kw_threshold.

    Secondary metric - must_retrieve (strict):
        Are specific function names present in retrieved function names?
        Reported but NOT the pass/fail criterion.
    """
    context = build_context(chunks[:top_k])
    retrieved_fns = [c.get("function_name", "") for c in chunks[:top_k]]
    retrieved_repos = [c.get("repo", "") for c in chunks[:top_k]]

    # Primary: keyword-in-context
    # Use context_keywords if defined, fall back to keywords
    kws = query_obj.get("context_keywords") or query_obj.get("keywords", [])
    kw_found = [kw for kw in kws if kw.lower() in context]
    kw_missed = [kw for kw in kws if kw.lower() not in context]

    if kws:
        kw_score = len(kw_found) / len(kws)
        kw_passed = kw_score >= kw_threshold
    else:
        kw_score = None
        kw_passed = None

    # Secondary: must_retrieve
    must = query_obj.get("must_retrieve", [])
    mr_hits = [fn for fn in must if fn in retrieved_fns]
    mr_misses = [fn for fn in must if fn not in retrieved_fns]
    mr_score = len(mr_hits) / len(must) if must else None

    # Anti-keywords (hallucination check, used in LLM eval later)
    anti = query_obj.get("anti_keywords", [])
    anti_hits = [kw for kw in anti if kw.lower() in context]

    return {
        "query_id": query_obj["id"],
        "tier": query_obj["tier"],
        "repos": query_obj["repos"],
        "query": query_obj["query"],
        # Primary
        "kw_score": kw_score,
        "kw_passed": kw_passed,
        "kw_found": kw_found,
        "kw_missed": kw_missed,
        # Secondary
        "mr_score": mr_score,
        "mr_hits": mr_hits,
        "mr_misses": mr_misses,
        # Metadata
        "anti_hits": anti_hits,
        "retrieved_functions": retrieved_fns,
        "retrieved_repos": retrieved_repos,
    }


def print_report(method, query_results, verbose=False, retrieval_details=None) -> dict:
    log.info("")
    log.info("=" * 70)
    log.info("Retrieval Evaluation - Method: %s  (keyword-in-context)", method.upper())
    log.info("=" * 70)

    verifiable = [r for r in query_results if r["kw_passed"] is not None]
    unverifiable = [r for r in query_results if r["kw_passed"] is None]
    passed = [r for r in verifiable if r["kw_passed"]]
    failed = [r for r in verifiable if not r["kw_passed"]]

    for tier in sorted(set(r["tier"] for r in query_results)):
        tv = [r for r in verifiable if r["tier"] == tier]
        if not tv:
            continue
        tp = sum(1 for r in tv if r["kw_passed"])
        avg = np.mean([r["kw_score"] for r in tv])
        log.info("  Tier %d: %d/%d passed (%.0f%%)  avg coverage %.2f",
                 tier, tp, len(tv), 100 * tp / len(tv), avg)

    log.info("")
    log.info("  Verifiable  : %d", len(verifiable))
    log.info("  Passed      : %d (%.1f%%)",
             len(passed), 100 * len(passed) / len(verifiable) if verifiable else 0)
    log.info("  Failed      : %d", len(failed))
    log.info("  Unverifiable: %d (no keywords - needs LLM judge)", len(unverifiable))

    if failed:
        log.info("")
        log.info("  Failed:")
        for r in failed:
            log.info("    [%s] T%d  score=%.2f  missed=%s",
                     r["query_id"], r["tier"], r["kw_score"] or 0, r["kw_missed"])

    if verbose and retrieval_details:
        log.info("")
        log.info("  Per-query chunks:")
        for qid, chunks in retrieval_details.items():
            r = next(x for x in query_results if x["query_id"] == qid)
            log.info("    %s  kw=%.2f  found=%s",
                     qid, r["kw_score"] or 0, r["kw_found"])
            for i, c in enumerate(chunks[:5], 1):
                log.info("      %d. [%.4f] %s::%s::%s",
                         i, c.get("_score", 0),
                         c.get("repo",""), c.get("filepath","")[:35],
                         c.get("function_name",""))

    avg_kw = np.mean([r["kw_score"] for r in verifiable
                      if r["kw_score"] is not None]) if verifiable else 0.0
    mr_v = [r for r in query_results if r["mr_score"] is not None]
    avg_mr = np.mean([r["mr_score"] for r in mr_v]) if mr_v else 0.0

    log.info("")
    log.info("  Avg keyword coverage : %.3f  (primary)", avg_kw)
    log.info("  Avg must_retrieve    : %.3f  (secondary)", avg_mr)
    log.info("  Pass rate            : %.1f%%",
             100 * len(passed) / len(verifiable) if verifiable else 0)
    log.info("=" * 70)

    return {
        "method": method,
        "total": len(query_results),
        "verifiable": len(verifiable),
        "passed": len(passed),
        "failed": len(failed),
        "unverifiable": len(unverifiable),
        "pass_rate": len(passed) / len(verifiable) if verifiable else 0.0,
        "avg_kw_score": float(avg_kw),
        "avg_mr_score": float(avg_mr),
    }


def save_results(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "retrieval_eval.json"
    with path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Saved to %s", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["bm25","dense","hybrid","all"], default="all")
    parser.add_argument("--tier", type=int, default=None)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--kw-threshold", type=float, default=0.5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    cfg = load_config()
    queries = load_test_queries(cfg["evaluation"]["test_queries_path"])
    if args.tier:
        queries = [q for q in queries if q["tier"] == args.tier]
    if args.query:
        queries = [q for q in queries if q["id"] == args.query]
    if not queries:
        log.error("No queries matched.")
        sys.exit(1)

    log.info("Eval: %d queries | top_k=%d | kw_threshold=%.1f",
             len(queries), args.top_k, args.kw_threshold)

    bm25, bm25_chunks = load_bm25(cfg)
    collection = load_chroma(cfg)
    model = load_embed_model(cfg)

    methods = ["bm25","dense","hybrid"] if args.method == "all" else [args.method]
    summaries = []
    output = {}

    for method in methods:
        log.info("\nRunning: %s", method)
        results = []
        details = {}
        t0 = time.time()

        for q in queries:
            if method == "bm25":
                chunks = retrieve_bm25(q["query"], bm25, bm25_chunks, args.top_k)
            elif method == "dense":
                chunks = retrieve_dense(q["query"], collection, model, args.top_k)
            else:
                chunks = retrieve_hybrid(q["query"], bm25, bm25_chunks,
                                          collection, model, cfg, args.top_k)
            results.append(evaluate_query(q, chunks, args.top_k, args.kw_threshold))
            details[q["id"]] = chunks

        dur = time.time() - t0
        log.info("  %.1fs (%.2fs/query)", dur, dur / len(queries))

        s = print_report(method, results, verbose=args.verbose,
                         retrieval_details=details if args.verbose else None)
        summaries.append(s)
        output[method] = {"summary": s, "per_query": results}

    if len(summaries) > 1:
        log.info("")
        log.info("=" * 70)
        log.info("Comparison")
        log.info("=" * 70)
        log.info("  %-10s  %8s  %10s  %12s  %10s",
                 "Method","Passed","Pass Rate","KW Coverage","MR Score")
        log.info("  " + "-"*55)
        for s in summaries:
            log.info("  %-10s  %8d  %9.1f%%  %11.3f  %9.3f",
                     s["method"], s["passed"],
                     100*s["pass_rate"], s["avg_kw_score"], s["avg_mr_score"])
        log.info("=" * 70)

    save_results(output, Path(cfg["evaluation"]["results_dir"]))
    log.info("\nNext: python eval/eval_baseline.py")


if __name__ == "__main__":
    main()