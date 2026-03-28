"""
eval/calibrate_test_cases.py

Helps verify and fix must_retrieve lists in test_queries.yaml.
For each query, searches the BM25 index for the function names
we expect to find, and shows what's actually available.

Usage:
    python eval/calibrate_test_cases.py              # check all queries
    python eval/calibrate_test_cases.py --query T1-004
    python eval/calibrate_test_cases.py --lookup minimize scipy
    python eval/calibrate_test_cases.py --search "groupby aggregation"
"""

import argparse
import pickle
import re
from pathlib import Path

import yaml

def load_bm25():
    path = Path("data/bm25_function.pkl")
    with path.open("rb") as f:
        payload = pickle.load(f)
    return payload["chunks"]

def load_queries():
    with open("eval/test_queries.yaml") as f:
        data = yaml.safe_load(f)
    return data["queries"]

def find_function(chunks: list[dict], name: str, repo: str = None) -> list[dict]:
    """Find chunks whose function_name matches (case-insensitive, partial ok)."""
    results = []
    name_lower = name.lower()
    for c in chunks:
        fn = c.get("function_name", "").lower()
        if name_lower in fn:
            if repo is None or c.get("repo", "") == repo:
                results.append(c)
    return results

def search_chunks(chunks: list[dict], query: str, top_k: int = 10) -> list[dict]:
    """Simple keyword search over function names + docstrings."""
    tokens = set(re.split(r"\W+", query.lower()))
    scores = []
    for c in chunks:
        text = f"{c.get('function_name','')} {c.get('docstring','')}".lower()
        score = sum(1 for t in tokens if t and t in text)
        if score > 0:
            scores.append((score, c))
    scores.sort(key=lambda x: -x[0])
    return [c for _, c in scores[:top_k]]

def check_query(q: dict, chunks: list[dict]) -> dict:
    """For a query, check which must_retrieve functions exist in the index."""
    must_retrieve = q.get("must_retrieve", [])
    repos = q.get("repos", [])
    
    found = []
    not_found = []
    alternatives = {}

    for fn in must_retrieve:
        # Search within the relevant repos first
        hits = []
        for repo in repos:
            hits = find_function(chunks, fn, repo=repo)
            if hits:
                break
        # Fallback: search all repos
        if not hits:
            hits = find_function(chunks, fn)
        
        if hits:
            found.append(fn)
        else:
            not_found.append(fn)
            # Suggest alternatives via keyword search
            alts = search_chunks(chunks, fn + " " + " ".join(repos), top_k=5)
            alternatives[fn] = [
                f"{c['repo']}::{c['function_name']}" for c in alts
            ]

    return {
        "query_id": q["id"],
        "tier": q["tier"],
        "found_in_index": found,
        "not_in_index": not_found,
        "alternatives": alternatives,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None,
                        help="Check a specific query ID")
    parser.add_argument("--lookup", nargs=2, metavar=("FUNCTION", "REPO"),
                        help="Look up a function name in a specific repo")
    parser.add_argument("--search", type=str, default=None,
                        help="Free-text search over function names + docstrings")
    parser.add_argument("--repo", type=str, default=None,
                        help="Filter --search to a specific repo")
    args = parser.parse_args()

    print("Loading BM25 index...")
    chunks = load_bm25()
    print(f"  {len(chunks):,} chunks loaded\n")

    # Mode 1: lookup a specific function in a repo
    if args.lookup:
        fn_name, repo = args.lookup
        results = find_function(chunks, fn_name, repo=repo)
        print(f"Functions matching '{fn_name}' in {repo}:")
        if results:
            for c in results[:20]:
                print(f"  {c['function_name']:40s}  {c['filepath']}")
        else:
            print(f"  [none found] - trying all repos...")
            results = find_function(chunks, fn_name)
            for c in results[:10]:
                print(f"  {c['repo']:15s}  {c['function_name']:40s}  {c['filepath']}")
        return

    # Mode 2: free-text search
    if args.search:
        query = args.search
        if args.repo:
            search_chunks_local = [c for c in chunks if c.get("repo") == args.repo]
        else:
            search_chunks_local = chunks
        results = search_chunks(search_chunks_local, query, top_k=15)
        print(f"Top results for '{query}':")
        for c in results:
            print(f"  {c['repo']:15s}  {c['function_name']:40s}  {c['filepath'][:50]}")
        return

    # Mode 3: check all queries (or a specific one)
    queries = load_queries()
    if args.query:
        queries = [q for q in queries if q["id"] == args.query]
        if not queries:
            print(f"Query {args.query} not found.")
            return

    print("=" * 70)
    print("Test Case Calibration Report")
    print("=" * 70)

    all_bad = []
    for q in queries:
        if not q.get("must_retrieve"):
            continue
        result = check_query(q, chunks)
        
        status = "OK" if not result["not_in_index"] else "FIX"
        print(f"\n[{status}] {result['query_id']} (Tier {result['tier']})")
        
        if result["found_in_index"]:
            print(f"  Found   : {result['found_in_index']}")
        
        if result["not_in_index"]:
            print(f"  Missing : {result['not_in_index']}")
            for fn, alts in result["alternatives"].items():
                print(f"    '{fn}' not found. Alternatives:")
                for alt in alts:
                    print(f"      - {alt}")
            all_bad.append(result["query_id"])

    print("\n" + "=" * 70)
    print(f"Summary: {len(all_bad)} queries have must_retrieve entries not in index")
    if all_bad:
        print(f"  Needs fixing: {all_bad}")
    print("\nUse --lookup <function> <repo> to investigate specific functions.")
    print("Use --search <keywords> --repo <repo> to find correct function names.")


if __name__ == "__main__":
    main()
