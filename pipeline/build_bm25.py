"""
pipeline/build_bm25.py

Builds a BM25 index over all chunks and serializes it to disk.
The BM25 index is the keyword-search component of our hybrid retrieval.

One index is built per chunking strategy (function / fixed / recursive).

Usage:
    python pipeline/build_bm25.py --strategy function
    python pipeline/build_bm25.py --strategy fixed
    python pipeline/build_bm25.py --strategy recursive
"""

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import yaml
from rank_bm25 import BM25Okapi
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def tokenize(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokenizer.
    Preserves underscores (important for function names like _wrapreduction).
    Lowercases everything.
    """
    import re
    # Split on whitespace and common code punctuation, but keep underscores
    tokens = re.split(r"[\s\(\)\[\]\{\}\.,;:\"'=\+\-\*/<>!@#\$%\^&\|\\`~]+", text.lower())
    return [t for t in tokens if len(t) > 1]

def build_text(chunk: dict, template: str) -> str:
    return template.format(
        function_name=chunk.get("function_name", ""),
        docstring=chunk.get("docstring", ""),
        raw_code=chunk.get("raw_code", ""),
        class_name=chunk.get("class_name", ""),
        filepath=chunk.get("filepath", ""),
        repo=chunk.get("repo", ""),
    ).strip()

def main() -> None:
    parser = argparse.ArgumentParser(description="Build BM25 index for hybrid retrieval")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["function", "fixed", "recursive"],
        required=True,
        help="Chunking strategy (required)",
    )
    args = parser.parse_args()

    cfg = load_config()
    chunking = args.strategy

    embed_cfg = cfg["embedding"]
    template = embed_cfg["text_template"]
    repo_names = cfg["repos"]["names"]

    # Output path - one BM25 index per chunking strategy
    bm25_path_template = cfg["bm25"]["index_path"]
    bm25_path = Path(bm25_path_template.format(chunking=chunking))
    bm25_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve chunks directory using the chunking strategy
    chunks_dir_template = cfg["repos"]["chunks_dir"]
    chunks_dir = Path(chunks_dir_template.format(chunking=chunking))

    log.info("=" * 60)
    log.info("Substrate - BM25 Index Builder")
    log.info("Strategy : %s", chunking)
    log.info("Output   : %s", bm25_path)
    log.info("=" * 60)

    # Load all chunks
    all_chunks = []
    for repo in repo_names:
        jsonl_path = chunks_dir / f"{repo}.jsonl"

        if not jsonl_path.exists():
            log.warning("Missing: %s - skipping", jsonl_path)
            continue

        count = 0
        with jsonl_path.open() as f:
            for line in f:
                all_chunks.append(json.loads(line))
                count += 1
        log.info("  Loaded %6d chunks from %s", count, jsonl_path.name)

    if not all_chunks:
        log.error("No chunks loaded. Run parse_functions.py first.")
        return

    log.info("\nTotal chunks: %d", len(all_chunks))

    # Build texts + tokenize
    log.info("Building texts and tokenizing...")
    t0 = time.time()

    texts = [build_text(c, template) for c in tqdm(all_chunks, desc="Building texts", unit="chunk", leave=False)]
    tokenized = [tokenize(t) for t in tqdm(texts, desc="Tokenizing", unit="chunk", leave=False)]

    log.info("Tokenized in %.1fs", time.time() - t0)

    # Build BM25
    log.info("Building BM25Okapi index...")
    t1 = time.time()
    bm25 = BM25Okapi(tokenized)
    log.info("BM25 index built in %.1fs", time.time() - t1)

    # Serialize
    log.info("Saving to %s...", bm25_path)
    payload = {
        "bm25": bm25,
        "chunks": all_chunks,
        "texts": texts,
        "tokenized": tokenized,
        "chunking": chunking,
    }
    with bm25_path.open("wb") as f:
        pickle.dump(payload, f)

    size_mb = bm25_path.stat().st_size / (1024 * 1024)
    log.info("- BM25 index saved: %.1f MB", size_mb)

    # Quick sanity check
    log.info("\nSanity check - querying 'numpy dtype float64':")
    test_tokens = tokenize("numpy dtype float64")
    scores = bm25.get_scores(test_tokens)
    top_idx = scores.argsort()[-5:][::-1]
    for idx in top_idx:
        c = all_chunks[idx]
        log.info(
            "  [%.3f] %s::%s::%s",
            scores[idx], c["repo"], c["filepath"], c["function_name"]
        )

    log.info("\nDone. Next step: python pipeline/eval_retrieval.py")

if __name__ == "__main__":
    main()
