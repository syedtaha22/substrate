"""
eval/eval_rag.py

RAG pipeline evaluation with LLM-as-Judge quality metrics.

Quality metrics (per-query, inline):

  faithfulness   - are answer claims grounded in retrieved code?
    1. LLM extracts atomic claims from the answer
    2. LLM verifies all claims in one batched call
    3. score = verified / total

  relevancy      - does the answer address the question?
    1. LLM generates 3 questions the answer would address
    2. Embed those + original query (all-MiniLM-L6-v2)
    3. score = mean cosine similarity

No external eval libraries. Ollama (local GPU) for LLM calls.

Usage:
    python eval/eval_rag.py --profile A5
    python eval/eval_rag.py --tier 1
    python eval/eval_rag.py
    python eval/eval_rag.py --no-judge    # keyword scoring only, fastest
    python eval/eval_rag.py --query T1-005
    python eval/eval_rag.py --dry-run
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import requests
import yaml
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.judge import LLMJudge
from app.retrieval import Retriever

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

def load_queries(path: str) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["queries"]

def load_baseline(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {q["query_id"]: q for q in data.get("per_query", [])}

# Context builder 
def build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        fp   = c.get("filepath", "?")
        fn   = c.get("function_name", "?")
        code = c.get("raw_code", c.get("_text", ""))[:700]
        parts.append(f"{fp}::{fn}\n```python\n{code}\n```")
    return "\n\n".join(parts)

# Keyword scoring 
def score_keywords(answer: str, keywords: list[str]) -> dict:
    if not answer or not keywords:
        return {"score": None, "found": [], "missed": []}
    al = answer.lower()
    found  = [k for k in keywords if k.lower() in al]
    missed = [k for k in keywords if k.lower() not in al]
    return {"score": len(found) / len(keywords), "found": found, "missed": missed}

# Main eval loop 
def run_eval(queries, retriever, system_prompt, gen_model, judge,
             embed_model, method, top_k, kw_threshold,
             run_judge, dry_run) -> list[dict]:

    results = []

    for i, q in enumerate(queries, 1):
        qid   = q["id"]
        query = q["query"]
        kws   = q.get("keywords", [])

        log.info("[%d/%d] %s - %s", i, len(queries), qid, query[:65])

        # Retrieve 
        t0     = time.time()
        chunks = retriever.retrieve(query, method=method, top_k=top_k)
        ret_t  = round(time.time() - t0, 2)

        repo_counts: dict[str, int] = {}
        for c in chunks:
            r = c.get("repo", "?")
            repo_counts[r] = repo_counts.get(r, 0) + 1
        log.info("  Retrieved: %s (%.2fs)",
                 ", ".join(f"{r}x{n}" for r, n in repo_counts.items()), ret_t)

        context = build_context(chunks)

        if dry_run:
            results.append({
                "query_id": qid, "tier": q["tier"], "query": query,
                "answer": "[DRY RUN]", "chunks_retrieved": len(chunks),
                "kw_score": None, "kw_passed": None,
                "kw_found": [], "kw_missed": kws,
                "faithfulness": None, "faithfulness_details": {},
                "relevancy": None, "relevancy_details": {},
                "ret_time_s": ret_t, "gen_time_s": 0, "method": method,
            })
            continue

        # Generate 
        user_msg = f"Retrieved source code:\n\n{context}\n\n---\n\nQuestion: {query}"
        t1     = time.time()
        answer = judge.call_ollama(
            [{"role": "system", "content": system_prompt},
             {"role": "user",   "content": user_msg}],
            model=gen_model, max_tokens=600, temp=0.1,
        )
        gen_t = round(time.time() - t1, 2)

        if answer is None:
            log.warning("  No answer - skipping")
            results.append({
                "query_id": qid, "tier": q["tier"], "query": query,
                "answer": None, "chunks_retrieved": len(chunks),
                "kw_score": None, "kw_passed": None,
                "kw_found": [], "kw_missed": kws,
                "faithfulness": None, "faithfulness_details": {},
                "relevancy": None, "relevancy_details": {},
                "ret_time_s": ret_t, "gen_time_s": gen_t, "method": method,
            })
            continue

        # Keyword score 
        kw      = score_keywords(answer, kws)
        kw_pass = (kw["score"] >= kw_threshold
                   if kw["score"] is not None else None)
        log.info("  KW: %.2f (%d/%d) - %s",
                 kw["score"] or 0, len(kw["found"]), len(kws),
                 "PASS" if kw_pass else "FAIL")

        # LLM-as-Judge (inline, per query) 
        faith_val, faith_details = None, {}
        relev_val, relev_details = None, {}
        if run_judge and chunks and embed_model is not None:
            faith_val, faith_details = judge.faithfulness(answer, context)
            relev_val, relev_details = judge.relevancy(query, answer, embed_model)
            if faith_val is not None:
                log.info("  Judge: faithfulness=%.3f  relevancy=%.3f",
                         faith_val, relev_val or 0)
            else:
                log.warning("  Judge: scoring failed - check debug logs")

        results.append({
            "query_id":         qid,
            "tier":             q["tier"],
            "query":            query,
            "answer":           answer,
            "chunks_retrieved": len(chunks),
            "repos_hit":        list(repo_counts.keys()),
            "kw_score":         kw["score"],
            "kw_passed":        kw_pass,
            "kw_found":         kw["found"],
            "kw_missed":        kw["missed"],
            "faithfulness":     faith_val,
            "faithfulness_details": faith_details,
            "relevancy":        relev_val,
            "relevancy_details": relev_details,
            "ret_time_s":       ret_t,
            "gen_time_s":       gen_t,
            "method":           method,
            "condition":        "rag",
        })

    return results

# Report 
def print_report(results: list[dict], baseline: dict) -> dict:
    log.info("")
    log.info("=" * 72)
    log.info("RAG Evaluation Results")
    log.info("=" * 72)

    scored     = [r for r in results if r.get("kw_score") is not None]
    passed     = [r for r in scored  if r.get("kw_passed")]
    judged     = [r for r in scored
                  if r.get("faithfulness") is not None
                  and r.get("relevancy")   is not None]

    for tier in sorted(set(r["tier"] for r in results)):
        tv = [r for r in scored if r["tier"] == tier]
        if not tv:
            continue
        tp     = sum(1 for r in tv if r.get("kw_passed"))
        avg    = float(np.mean([r["kw_score"] for r in tv]))
        bl_sc  = [baseline[r["query_id"]]["score"]
                  for r in tv
                  if r["query_id"] in baseline
                  and baseline[r["query_id"]].get("score") is not None]
        bl_avg = float(np.mean(bl_sc)) if bl_sc else None
        delta  = (avg - bl_avg) if bl_avg is not None else None
        extra  = (f"  baseline {bl_avg:.3f}  {delta:+.3f}"
                  if delta is not None else "")
        log.info("  Tier %d: %d/%d passed (%.0f%%)  KW %.3f%s",
                 tier, tp, len(tv), 100 * tp / len(tv), avg, extra)

    log.info("")
    rag_mean = float(np.mean([r["kw_score"] for r in scored])) if scored else 0
    log.info("  Queries answered : %d / %d", len(scored), len(results))
    log.info("  Passed           : %d (%.1f%%)",
             len(passed), 100 * len(passed) / len(scored) if scored else 0)
    log.info("  Avg KW score     : %.3f", rag_mean)

    bl_all = [baseline[r["query_id"]]["score"]
               for r in scored
               if r["query_id"] in baseline
               and baseline[r["query_id"]].get("score") is not None]
    if bl_all:
        bl_mean = float(np.mean(bl_all))
        log.info("  Baseline avg     : %.3f", bl_mean)
        log.info("  RAG improvement  : %+.3f  (%.1f%% relative)",
                 rag_mean - bl_mean,
                 100 * (rag_mean - bl_mean) / bl_mean if bl_mean else 0)

    avg_f = avg_r = None
    if judged:
        avg_f = float(np.mean([r["faithfulness"] for r in judged]))
        avg_r = float(np.mean([r["relevancy"]    for r in judged]))
        log.info("")
        log.info("  LLM-as-Judge (n=%d):", len(judged))
        log.info("    Faithfulness (claim-based) : %.3f", avg_f)
        log.info("    Relevancy    (cosine sim)  : %.3f", avg_r)

    log.info("")
    ret_t = [r["ret_time_s"] for r in results if r.get("ret_time_s")]
    gen_t = [r["gen_time_s"] for r in results if r.get("gen_time_s")]
    if ret_t: log.info("  Avg retrieval time : %.2fs", np.mean(ret_t))
    if gen_t: log.info("  Avg gen time       : %.2fs", np.mean(gen_t))
    log.info("=" * 72)

    return {
        "condition":        "rag",
        "total":            len(results),
        "scored":           len(scored),
        "passed":           len(passed),
        "pass_rate":        len(passed) / len(scored) if scored else 0,
        "avg_kw_score":     rag_mean,
        "avg_faithfulness": avg_f,
        "avg_relevancy":    avg_r,
        "judge_n":          len(judged),
        "avg_ret_time_s":   float(np.mean(ret_t)) if ret_t else None,
        "avg_gen_time_s":   float(np.mean(gen_t)) if gen_t else None,
    }

def save_results(summary: dict, per_query: list[dict], output_dir: Path, profile: str = "A5"):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"rag_eval_{profile.lower()}.json"
    with path.open("w") as f:
        json.dump({"summary": summary, "per_query": per_query}, f, indent=2)
    log.info("Saved -> %s", path)

# Main 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, choices=["A1", "A2", "A3", "A4", "A5"], default="A5", help="Ablation profile (default: A5)")
    parser.add_argument("--method",    default="hybrid", choices=["hybrid", "dense", "bm25"])
    parser.add_argument("--top-k",    type=int,   default=5)
    parser.add_argument("--tier",     type=int,   default=None)
    parser.add_argument("--query",    type=str,   default=None)
    parser.add_argument("--threshold",type=float, default=0.4)
    parser.add_argument("--no-judge", action="store_true", help="Keyword scoring only - skip LLM-as-Judge")
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args()

    cfg           = load_config()
    profile       = cfg["profiles"][args.profile]
    strategy      = profile["chunking"]  # e.g., "function", "fixed", "recursive"
    gen_cfg       = cfg["generation"]
    system_prompt = gen_cfg["system_prompt"].strip()
    gen_model     = gen_cfg["model"]
    judge_model   = cfg["evaluation"].get("judge", gen_model)
    output_dir    = Path(cfg["evaluation"]["results_dir"])

    queries = load_queries(cfg["evaluation"]["test_queries_path"])
    if args.tier:  queries = [q for q in queries if q["tier"] == args.tier]
    if args.query: queries = [q for q in queries if q["id"] == args.query]
    if not queries:
        log.error("No queries matched")
        sys.exit(1)

    baseline_path = output_dir / "baseline.json"
    baseline = {}
    if baseline_path.exists():
        baseline = load_baseline(str(baseline_path))
        log.info("Loaded baseline: %d queries", len(baseline))
    else:
        log.warning("No baseline.json - run eval_baseline.py first")

    if not args.dry_run:
        try:
            requests.get("http://localhost:11434", timeout=3)
        except Exception:
            log.error("Ollama not reachable. Run: ollama serve")
            sys.exit(1)

    # Load embed model for relevancy scoring
    embed_model = None
    if not args.no_judge and not args.dry_run:
        emb_name = cfg["embedding"]["model"]
        log.info("Loading embed model: %s", emb_name)
        embed_model = SentenceTransformer(emb_name)

    log.info("=" * 72)
    log.info("Substrate - RAG Evaluation")
    log.info("Profile: %s  (%s chunking + %s retrieval)", args.profile, profile["chunking"], profile["retrieval"])
    log.info("Model  : %s  (judge: %s)", gen_model, judge_model)
    log.info("Method : %s  top-k=%d", args.method, args.top_k)
    log.info("Queries: %d  llm-as-judge: %s", len(queries), not args.no_judge)
    log.info("=" * 72)

    retriever = Retriever(strategy=strategy)
    retriever.load()

    # Initialize LLM judge (embedding model passed separately to relevancy method)
    judge = LLMJudge(judge_model=judge_model)

    per_query = run_eval(
        queries=queries,
        retriever=retriever,
        system_prompt=system_prompt,
        gen_model=gen_model,
        judge=judge,
        embed_model=embed_model,
        method=args.method,
        top_k=args.top_k,
        kw_threshold=args.threshold,
        run_judge=not args.no_judge,
        dry_run=args.dry_run,
    )

    summary = print_report(per_query, baseline)
    save_results(summary, per_query, output_dir, args.profile)
    log.info("Done.")


if __name__ == "__main__":
    main()