"""
pipeline/parse_functions.py

Parses all cloned repositories using tree-sitter and extracts
function-level chunks - the core research contribution of Substrate.

Each chunk contains:
  - raw source code of the function
  - structured metadata: repo, filepath, function name, class context,
    docstring, line numbers, language

Output: data/chunks/{repo_name}.jsonl  (one JSON object per line)

Usage:
    python pipeline/parse_functions.py --repo numpy
    python pipeline/parse_functions.py              # all repos
    python pipeline/parse_functions.py --stats      # print stats only
"""

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from tree_sitter import Language, Parser
import tree_sitter_python as tspython
from tqdm import tqdm

# Configuration 
REPOS_DIR   = Path("data/repos")
CHUNKS_DIR  = Path("data/chunks")

# Repos in dependency order (bottom of stack -> top)
REPO_NAMES = ["numpy", "scipy", "pandas", "scikit-learn", "pytorch", "transformers"]

# We skip files in these directories - they add noise without signal
SKIP_DIRS = {
    "test", "tests", "testing",
    "benchmarks", "bench",
    "doc", "docs", "documentation",
    "examples", "tutorials",
    "build", "dist", "_build",
    "__pycache__",
}

# Functions shorter than this are usually boilerplate (getters, __repr__, etc.)
MIN_LINES = 3

# Functions longer than this are usually auto-generated or data files
MAX_LINES = 500

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Data model 
@dataclass
class FunctionChunk:
    # Identity
    chunk_id:      str    # "{repo}::{filepath}::{function_name}::{line_start}"
    repo:          str
    filepath:      str    # relative to repo root
    language:      str    # "python"

    # Function metadata
    function_name: str
    class_name:    str    # empty string if top-level function
    is_method:     bool

    # Content
    docstring:     str    # empty string if none
    raw_code:      str    # full source text of the function

    # Location
    line_start:    int
    line_end:      int
    line_count:    int


# Tree-sitter setup 
def build_parser() -> Parser:
    """Build a tree-sitter parser for Python."""
    PY_LANGUAGE = Language(tspython.language(), "python")
    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    return parser


# Extraction logic 
def get_node_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

def extract_docstring(function_node, source_bytes: bytes) -> str:
    """
    Extract the docstring from a function_definition node if present.
    The docstring is the first expression_statement containing a string
    inside the function body.
    """
    body = next(
        (child for child in function_node.children if child.type == "block"),
        None,
    )
    if body is None:
        return ""

    for child in body.children:
        if child.type == "expression_statement":
            for subchild in child.children:
                if subchild.type in ("string", "concatenated_string"):
                    raw = get_node_text(subchild, source_bytes)
                    # Strip quotes and clean up
                    return raw.strip('"""').strip("'''").strip('"').strip("'").strip()
        # Docstring must be the very first statement
        if child.type not in ("comment", "expression_statement"):
            break

    return ""


def extract_functions_from_file(
    filepath: Path,
    repo_name: str,
    parser: Parser,
) -> list[FunctionChunk]:
    """
    Parse a single Python file and extract all function/method definitions
    as FunctionChunk objects.
    """
    try:
        source_bytes = filepath.read_bytes()
    except (OSError, PermissionError):
        return []

    try:
        tree = parser.parse(source_bytes)
    except Exception:
        return []

    chunks: list[FunctionChunk] = []
    repo_root = REPOS_DIR / repo_name
    rel_path = str(filepath.relative_to(repo_root))

    def walk(node, class_name: str = "") -> None:
        """Recursively walk the AST, tracking class context."""

        if node.type == "class_definition":
            # Extract class name for methods defined inside
            name_node = next(
                (c for c in node.children if c.type == "identifier"), None
            )
            current_class = get_node_text(name_node, source_bytes) if name_node else ""
            for child in node.children:
                walk(child, class_name=current_class)
            return

        if node.type == "function_definition":
            name_node = next(
                (c for c in node.children if c.type == "identifier"), None
            )
            if name_node is None:
                return

            func_name = get_node_text(name_node, source_bytes)
            raw_code  = get_node_text(node, source_bytes)
            line_start = node.start_point[0] + 1  # 1-indexed
            line_end   = node.end_point[0] + 1
            line_count = line_end - line_start + 1

            # Apply filters
            if line_count < MIN_LINES or line_count > MAX_LINES:
                # Still recurse - nested functions inside valid ones are fine
                for child in node.children:
                    walk(child, class_name=class_name)
                return

            docstring = extract_docstring(node, source_bytes)

            chunk_id = f"{repo_name}::{rel_path}::{func_name}::{line_start}"

            chunk = FunctionChunk(
                chunk_id=chunk_id,
                repo=repo_name,
                filepath=rel_path,
                language="python",
                function_name=func_name,
                class_name=class_name,
                is_method=bool(class_name),
                docstring=docstring,
                raw_code=raw_code,
                line_start=line_start,
                line_end=line_end,
                line_count=line_count,
            )
            chunks.append(chunk)

            # Recurse into nested functions
            for child in node.children:
                walk(child, class_name=class_name)
            return

        # For all other node types, just recurse
        for child in node.children:
            walk(child, class_name=class_name)

    walk(tree.root_node)
    return chunks

def should_skip(filepath: Path) -> bool:
    """Return True if this file is in a directory we want to skip."""
    parts = set(filepath.parts)
    return bool(parts & SKIP_DIRS)

# Per-repo pipeline 
def parse_repo(repo_name: str, parser: Parser) -> dict:
    """
    Parse all Python files in a repo and write chunks to JSONL.
    Returns stats dict.
    """
    repo_dir = REPOS_DIR / repo_name
    if not repo_dir.exists():
        log.error("Repo not found: %s - run clone_repos.py first", repo_dir)
        return {}

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHUNKS_DIR / f"{repo_name}.jsonl"

    py_files = [
        f for f in repo_dir.rglob("*.py")
        if not should_skip(f)
    ]

    log.info("  %d .py files to parse (after skip filtering)", len(py_files))

    total_chunks = 0
    total_files_with_chunks = 0
    skipped_files = 0
    t0 = time.time()

    with out_path.open("w", encoding="utf-8") as fout:
        for filepath in tqdm(py_files, desc=f"  {repo_name}", unit="file", leave=False):
            chunks = extract_functions_from_file(filepath, repo_name, parser)
            if chunks:
                total_files_with_chunks += 1
                for chunk in chunks:
                    fout.write(json.dumps(asdict(chunk)) + "\n")
                    total_chunks += 1
            else:
                skipped_files += 1

    duration = time.time() - t0
    size_mb = out_path.stat().st_size / (1024 * 1024)

    stats = {
        "repo": repo_name,
        "py_files": len(py_files),
        "files_with_functions": total_files_with_chunks,
        "total_chunks": total_chunks,
        "output_path": str(out_path),
        "output_mb": round(size_mb, 1),
        "duration_s": round(duration, 1),
    }

    log.info(
        "    %d chunks from %d files to %s (%.1f MB) in %.1fs",
        total_chunks, total_files_with_chunks,
        out_path.name, size_mb, duration,
    )
    return stats

# Stats display 
def print_summary(all_stats: list[dict]) -> None:
    log.info("")
    log.info("=" * 65)
    log.info("%-16s  %8s  %8s  %10s  %6s", "Repo", "PY files", "Chunks", "JSONL (MB)", "Time")
    log.info("-" * 65)
    total_chunks = 0
    for s in all_stats:
        log.info(
            "%-16s  %8d  %8d  %10.1f  %5.1fs",
            s["repo"], s["py_files"], s["total_chunks"], s["output_mb"], s["duration_s"],
        )
        total_chunks += s["total_chunks"]
    log.info("-" * 65)
    log.info("%-16s  %8s  %8d", "TOTAL", "", total_chunks)
    log.info("=" * 65)
    log.info("")
    log.info("Chunks written to: %s/", CHUNKS_DIR)
    log.info("")

    # Warn if total is close to Pinecone free tier limit
    if total_chunks > 80_000:
        log.warning(
            "Total chunks (%d) exceeds Pinecone free tier (100k).", total_chunks
        )
        log.warning(
            "We will sample to 80k during embed_and_upsert.py."
        )
    else:
        log.info("Total chunks (%d) fits within Pinecone free tier (100k).", total_chunks)

    log.info("Next step: python pipeline/embed_and_upsert.py")

def print_sample_chunks(repo_name: str, n: int = 3) -> None:
    """Print a few sample chunks so we can visually verify quality."""
    out_path = CHUNKS_DIR / f"{repo_name}.jsonl"
    if not out_path.exists():
        return

    log.info("")
    log.info("Sample chunks from %s:", repo_name)
    log.info("-" * 65)

    with out_path.open() as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            chunk = json.loads(line)
            log.info(
                "[%d] %s::%s  (lines %d–%d, %d loc)",
                i + 1,
                chunk["filepath"],
                chunk["function_name"],
                chunk["line_start"],
                chunk["line_end"],
                chunk["line_count"],
            )
            if chunk["docstring"]:
                doc_preview = chunk["docstring"][:120].replace("\n", " ")
                log.info("     docstring: %s...", doc_preview)
            code_preview = chunk["raw_code"][:200].replace("\n", "↵ ")
            log.info("     code:      %s...", code_preview)
            log.info("")

# Entry point 
def main() -> None:
    parser_arg = argparse.ArgumentParser(
        description="Parse cloned repos into function-level chunks"
    )
    parser_arg.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Parse a single repo (e.g. --repo numpy). Default: all repos.",
    )
    parser_arg.add_argument(
        "--stats",
        action="store_true",
        help="Print stats for already-parsed chunks without re-parsing.",
    )
    parser_arg.add_argument(
        "--sample",
        action="store_true",
        help="Print sample chunks after parsing for quality inspection.",
    )
    args = parser_arg.parse_args()

    repos = [args.repo] if args.repo else REPO_NAMES

    # Stats-only mode
    if args.stats:
        all_stats = []
        for repo_name in repos:
            out_path = CHUNKS_DIR / f"{repo_name}.jsonl"
            if not out_path.exists():
                log.warning("No chunks file for %s - parse it first.", repo_name)
                continue
            count = sum(1 for _ in out_path.open())
            size_mb = out_path.stat().st_size / (1024 * 1024)
            all_stats.append({
                "repo": repo_name,
                "py_files": 0,
                "total_chunks": count,
                "output_mb": round(size_mb, 1),
                "duration_s": 0,
            })
        print_summary(all_stats)
        return

    # Validate repos exist
    for repo_name in repos:
        if not (REPOS_DIR / repo_name).exists():
            log.error(
                "Repo '%s' not found in %s. Run clone_repos.py first.",
                repo_name, REPOS_DIR,
            )
            return

    log.info("=" * 65)
    log.info("Substrate - Function Parser")
    log.info("Repos   : %s", ", ".join(repos))
    log.info("Output  : %s/", CHUNKS_DIR)
    log.info("=" * 65)

    ts_parser = build_parser()
    all_stats = []

    for repo_name in repos:
        log.info("")
        log.info("Parsing %s...", repo_name)
        stats = parse_repo(repo_name, ts_parser)
        if stats:
            all_stats.append(stats)
            if args.sample:
                print_sample_chunks(repo_name, n=2)

    if all_stats:
        print_summary(all_stats)

if __name__ == "__main__":
    main()
