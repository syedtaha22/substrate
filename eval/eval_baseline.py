"""
eval/eval_baseline.py

Tests raw LLM with NO retrieval context.
Establishes the floor that all RAG configurations must beat.

Scoring:
  - For each query, call LLM with system prompt only (no retrieved chunks)
  - Score = % of query's `keywords` found in the generated answer
  - Pass if keyword coverage >= threshold (default 0.4 — slightly lower than
    retrieval eval since generation may paraphrase rather than use exact terms)

Output: eval/results/baseline.json

Setup:
  ollama pull llama2           # or mistral, llama3.1, etc.
  ollama serve                 # start local server at localhost:11434

Usage:
    python eval/eval_baseline.py
    python eval/eval_baseline.py --tier 1            # single tier
    python eval/eval_baseline.py --query T1-001      # single query
    python eval/eval_baseline.py --model llama2      # override model
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

sys.path.insert(0, str(Path(__file__).parent.parent))

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

def load_test_queries(path: str) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["queries"]

# Prompts 

# System prompt loaded from config.yaml -> generation.system_prompt
# Edit there, not here.
def _load_system_prompt(cfg: dict) -> str:
    # Try to get system_prompt
    # If not found throw error and exit. instead of silently continuing
    try :
        return cfg["generation"]["system_prompt"]
    except KeyError:
        log.error("System prompt not found in config.yaml under generation.system_prompt")
        sys.exit(1)

# Instruction template — no context injected (baseline condition)
BASELINE_TEMPLATE = """Question: {query}

Answer:"""

# Ollama (local LLM)
def call_ollama(
    prompt: str,
    model: str,
    system_prompt: str = 'You are an expert software engineer.',
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> str | None:
    """
    Call local Ollama server (http://localhost:11434).
    Install: curl -fsSL https://ollama.ai/install.sh | sh
    Pull model: ollama pull llama3.1  (or mistral, llama3.1, phi3, etc.)
    Start: ollama serve
    """
    try:
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()["message"]["content"].strip()
        log.error("Ollama error %d: %s", resp.status_code, resp.text[:200])
        return None
    except requests.exceptions.ConnectionError:
        log.error(
            "Cannot connect to Ollama at localhost:11434. "
            "Is it running? Start with: ollama serve"
        )
        sys.exit(1)
    except Exception as e:
        log.error("Ollama request error: %s", e)
        return None


# Scoring 
def score_answer(answer: str, keywords: list[str]) -> dict:
    """
    Check what % of expected keywords appear in the generated answer.
    Uses the `keywords` field (LLM answer eval) not context_keywords.
    Case-insensitive, partial match allowed.
    """
    if not answer or not keywords:
        return {"score": None, "found": [], "missed": [], "passed": None}

    answer_lower = answer.lower()
    found = [kw for kw in keywords if kw.lower() in answer_lower]
    missed = [kw for kw in keywords if kw.lower() not in answer_lower]
    score = len(found) / len(keywords)

    return {
        "score": score,
        "found": found,
        "missed": missed,
    }

# Main eval loop 
def run_baseline(
    queries: list[dict],
    model: str,
    pass_threshold: float,
    cfg: dict = None,
) -> list[dict]:
    results = []
    total = len(queries)

    for i, q in enumerate(queries, 1):
        qid = q["id"]
        log.info("[%d/%d] %s — %s...", i, total, qid, q["query"][:60])

        prompt = BASELINE_TEMPLATE.format(query=q["query"])

        t0 = time.time()
        system_prompt = _load_system_prompt(cfg or {})
        answer = call_ollama(prompt, model, system_prompt=system_prompt)
        duration = time.time() - t0

        if answer is None:
            log.warning("  No answer returned — skipping")
            results.append({
                "query_id": qid,
                "tier": q["tier"],
                "query": q["query"],
                "answer": None,
                "score": None,
                "passed": None,
                "found": [],
                "missed": q.get("keywords", []),
                "duration_s": duration,
                "model": model,
                "condition": "baseline_no_rag",
            })
            continue

        log.info("  Answer (%d chars, %.1fs): %s...",
                 len(answer), duration, answer[:80].replace("\n", " "))

        score_result = score_answer(answer, q.get("keywords", []))
        passed = (
            score_result["score"] >= pass_threshold
            if score_result["score"] is not None else None
        )
        score_result["passed"] = passed

        log.info("  Score: %.2f (%d/%d keywords) — %s",
                 score_result["score"] or 0,
                 len(score_result["found"]),
                 len(q.get("keywords", [])),
                 "PASS" if passed else "FAIL")

        results.append({
            "query_id": qid,
            "tier": q["tier"],
            "query": q["query"],
            "answer": answer,
            "score": score_result.get("score"),
            "passed": score_result.get("passed"),
            "found": score_result.get("found", []),
            "missed": score_result.get("missed", []),
            "duration_s": duration if not dry_run else 0,
            "model": model,
            "condition": "baseline_no_rag",
        })

    return results


# Report 

def print_report(results: list[dict], pass_threshold: float) -> dict:
    log.info("")
    log.info("=" * 70)
    log.info("Baseline Evaluation — No RAG")
    log.info("=" * 70)

    scored = [r for r in results if r["score"] is not None]
    passed = [r for r in scored if r.get("passed")]
    failed = [r for r in scored if not r.get("passed")]
    skipped = [r for r in results if r["score"] is None]

    # Per-tier
    for tier in sorted(set(r["tier"] for r in results)):
        tv = [r for r in scored if r["tier"] == tier]
        if not tv:
            continue
        tp = sum(1 for r in tv if r.get("passed"))
        avg = np.mean([r["score"] for r in tv])
        log.info("  Tier %d: %d/%d passed (%.0f%%)  avg kw score %.2f",
                 tier, tp, len(tv), 100 * tp / len(tv) if tv else 0, avg)

    log.info("")
    log.info("  Total queries  : %d", len(results))
    log.info("  Scored         : %d", len(scored))
    log.info("  Passed         : %d (%.1f%%)",
             len(passed), 100 * len(passed) / len(scored) if scored else 0)
    log.info("  Failed         : %d", len(failed))
    log.info("  Skipped (error): %d", len(skipped))

    if scored:
        avg_score = np.mean([r["score"] for r in scored])
        log.info("")
        log.info("  Avg keyword score : %.3f  (baseline — no RAG)", avg_score)
        log.info("  Pass threshold    : %.1f", pass_threshold)
        log.info("")

    log.info("=" * 70)

    return {
        "condition": "baseline_no_rag",
        "total": len(results),
        "scored": len(scored),
        "passed": len(passed),
        "failed": len(failed),
        "skipped": len(skipped),
        "pass_rate": len(passed) / len(scored) if scored else 0.0,
        "avg_score": float(np.mean([r["score"] for r in scored])) if scored else 0.0,
        "pass_threshold": pass_threshold,
    }


def save_results(summary: dict, per_query: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "baseline.json"
    with path.open("w") as f:
        json.dump({"summary": summary, "per_query": per_query}, f, indent=2)
    log.info("Results saved to %s", path)


# Main 
def main() -> None:
    load_dotenv()

    cfg = load_config()
    default_model = cfg["generation"]["model"]

    parser = argparse.ArgumentParser(description="Baseline eval — raw LLM, no RAG (Ollama)")
    parser.add_argument("--model", type=str, default=default_model,
                        help=f"Ollama model ID (default: {default_model} from config.yaml)")
    parser.add_argument("--tier", type=int, default=None)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Keyword coverage pass threshold (default 0.4)")
    args = parser.parse_args()

    queries = load_test_queries(cfg["evaluation"]["test_queries_path"])

    if args.tier:
        queries = [q for q in queries if q["tier"] == args.tier]
    if args.query:
        queries = [q for q in queries if q["id"] == args.query]
    if not queries:
        log.error("No queries matched.")
        sys.exit(1)

    log.info("=" * 70)
    log.info("Substrate — Baseline Evaluation (No RAG)")
    log.info("Model     : %s", args.model)
    log.info("Queries   : %d", len(queries))
    log.info("Threshold : %.1f", args.threshold)
    log.info("=" * 70)

    per_query = run_baseline(
        queries, args.model,
        pass_threshold=args.threshold,
        cfg=cfg,
    )

    summary = print_report(per_query, args.threshold)
    save_results(summary, per_query, Path(cfg["evaluation"]["results_dir"]))


if __name__ == "__main__":
    main()