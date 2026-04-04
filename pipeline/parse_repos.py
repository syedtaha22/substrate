"""
pipeline/parse_repos.py

Unified parser for all chunking strategies.
Handles function-level, fixed-token, and recursive-character chunking.

Strategies:
  function  — extract function/method definitions (A3, A5)
  fixed     — split every file into fixed-size token windows (A1, A4)
  recursive — split on class/def/blank boundaries, then by size (A2)

Each chunk contains:
  - raw source code of the function
  - structured metadata: repo, filepath, function name, class context,
    docstring, line numbers, language

Output: data/chunks_{strategy}/{repo}.jsonl

Usage:
    python pipeline/parse_repos.py --strategy function
    python pipeline/parse_repos.py --strategy fixed --repo numpy
    python pipeline/parse_repos.py --strategy recursive --stats
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
import yaml

# Configuration 
REPOS_DIR   = Path("data/repos")

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

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def should_skip(filepath: Path) -> bool:
    """Return True if this file is in a directory we want to skip."""
    parts = set(filepath.parts)
    return bool(parts & SKIP_DIRS)

# Data model 
@dataclass
class Chunk:
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

    # Chunking strategy (for non-function chunks)
    chunk_strategy: str = ""

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

# Tree-sitter setup 
def build_parser() -> Parser:
    """Build a tree-sitter parser for Python."""
    PY_LANGUAGE = Language(tspython.language(), "python")
    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    return parser

def extract_functions_from_file(
    filepath: Path,
    repo_name: str,
    parser: Parser,
    min_lines: int,
    max_lines: int,
) -> list[Chunk]:
    """
    Parse a single Python file and extract all function/method definitions
    as Chunk objects.
    """
    try:
        source_bytes = filepath.read_bytes()
        tree = parser.parse(source_bytes)
    except Exception as e:
        log.error("Failed to parse %s: %s", filepath, e)
        return []

    chunks: list[Chunk] = []
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

            func_name  = get_node_text(name_node, source_bytes)
            raw_code   = get_node_text(node, source_bytes)
            line_start = node.start_point[0] + 1  # 1-indexed
            line_end   = node.end_point[0] + 1
            line_count = line_end - line_start + 1

            # Apply filters
            if line_count < min_lines or line_count > max_lines:
                # Still recurse - nested functions inside valid ones are fine
                for child in node.children:
                    walk(child, class_name=class_name)
                return

            docstring = extract_docstring(node, source_bytes)

            chunk_id = f"{repo_name}::{rel_path}::{func_name}::{line_start}"

            chunk = Chunk(
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
                chunk_strategy="function",
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

# Fixed-size chunking (Used for A1, A4)
def chunk_fixed(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int, str]]:
    """
    Split text into fixed-token-count windows.
    Uses word-level approximation (1 word ≈ 1.3 tokens).
    Returns list of (start_line, end_line, text).
    """
    lines = text.splitlines(keepends=True)
    approx_tokens_per_line = 8
    lines_per_chunk = max(10, chunk_size // approx_tokens_per_line)
    overlap_lines = max(1, chunk_overlap // approx_tokens_per_line)

    chunks = []
    i = 0
    while i < len(lines):
        end = min(i + lines_per_chunk, len(lines))
        chunk_text = "".join(lines[i:end])
        if chunk_text.strip():
            chunks.append((i + 1, end, chunk_text))
        i += lines_per_chunk - overlap_lines

    return chunks

# Recursive character chunking (Used for A2)
def chunk_recursive(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> list[tuple[int, int, str]]:
    """
    Split text on separators in priority order, then by size.
    Similar to LangChain's RecursiveCharacterTextSplitter logic.
    Returns list of (start_line, end_line, text).
    """
    lines = text.splitlines(keepends=True)
    approx_chars_per_chunk = chunk_size * 4

    result = []
    current_start_line = 0
    current_chunk_lines = []
    current_size = 0

    for line_idx, line in enumerate(lines):
        line_len = len(line)

        is_separator = any(
            sep.lstrip("\n") and line.lstrip().startswith(sep.lstrip("\n"))
            for sep in separators
            if sep.strip()
        )

        if (is_separator or current_size + line_len > approx_chars_per_chunk) and current_chunk_lines:
            chunk_text = "".join(current_chunk_lines)
            if chunk_text.strip():
                result.append((current_start_line + 1, line_idx, chunk_text))

            overlap_count = max(1, chunk_overlap // 40)
            overlap_lines = current_chunk_lines[-overlap_count:]
            overlap_start = line_idx - len(overlap_lines)
            current_start_line = max(0, overlap_start)
            current_chunk_lines = list(overlap_lines)
            current_size = sum(len(l) for l in current_chunk_lines)

        current_chunk_lines.append(line)
        current_size += line_len

    if current_chunk_lines:
        chunk_text = "".join(current_chunk_lines)
        if chunk_text.strip():
            result.append((current_start_line + 1, len(lines), chunk_text))

    return result

# Per-repo pipeline - Function-level extraction
def parse_functions(repo_name: str, cfg: dict, ts_parser: Parser) -> dict:
    """
    Parse a repo using function-level extraction (tree-sitter).
    Returns stats dict.
    """
    repo_dir = REPOS_DIR / repo_name
    if not repo_dir.exists():
        log.error("Repo not found: %s - run clone_repos.py first", repo_dir)
        return {}

    output_dir = Path("data/chunks_function")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{repo_name}.jsonl"

    py_files = [f for f in repo_dir.rglob("*.py") if not should_skip(f)]
    log.info("  %d .py files to parse (strategy=function)", len(py_files))

    total_chunks = 0
    total_files_with_chunks = 0
    skipped_files = 0
    t0 = time.time()
    chunking_cfg = cfg["chunking"]["function"]

    with out_path.open("w", encoding="utf-8") as fout:
        for filepath in tqdm(py_files, desc=f"  {repo_name}", unit="file", leave=False):
            chunks = extract_functions_from_file(
                filepath, repo_name, ts_parser,
                min_lines=chunking_cfg.get("min_lines", 3),
                max_lines=chunking_cfg.get("max_lines", 500),
            )
            if chunks:
                total_files_with_chunks += 1
                for chunk in chunks:
                    fout.write(json.dumps(asdict(chunk)) + "\n")
                    total_chunks += 1
            else:
                skipped_files += 1

    duration = time.time() - t0
    size_mb = out_path.stat().st_size / (1024 * 1024)

    return {
        "repo": repo_name,
        "py_files": len(py_files),
        "strategy": "function",
        "total_chunks": total_chunks,
        "output_path": str(out_path),
        "output_mb": round(size_mb, 1),
        "duration_s": round(duration, 1),
    }


# Per-repo pipeline - Fixed-size chunking
def parse_fixed(repo_name: str, cfg: dict) -> dict:
    """
    Parse a repo using fixed-token-size chunking.
    Returns stats dict.
    """
    repo_dir = REPOS_DIR / repo_name
    if not repo_dir.exists():
        log.error("Repo not found: %s - run clone_repos.py first", repo_dir)
        return {}

    output_dir = Path("data/chunks_fixed")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{repo_name}.jsonl"

    py_files = [f for f in repo_dir.rglob("*.py") if not should_skip(f)]
    log.info("  %d .py files to parse (strategy=fixed)", len(py_files))

    total_chunks = 0
    t0 = time.time()
    chunking_cfg = cfg["chunking"]["fixed"]
    chunk_size = chunking_cfg.get("chunk_size", 512)
    chunk_overlap = chunking_cfg.get("chunk_overlap", 64)

    with out_path.open("w", encoding="utf-8") as fout:
        for filepath in py_files:
            try:
                text = filepath.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            repo_root = REPOS_DIR / repo_name
            rel_path = str(filepath.relative_to(repo_root))
            raw_chunks = chunk_fixed(text, chunk_size, chunk_overlap)

            for line_start, line_end, chunk_text in raw_chunks:
                if not chunk_text.strip():
                    continue
                line_count = line_end - line_start + 1
                chunk_id = f"{repo_name}::{rel_path}::fixed::{line_start}"

                chunk = Chunk(
                    chunk_id=chunk_id,
                    repo=repo_name,
                    filepath=rel_path,
                    language="python",
                    function_name="",
                    class_name="",
                    is_method=False,
                    docstring="",
                    raw_code=chunk_text,
                    line_start=line_start,
                    line_end=line_end,
                    line_count=line_count,
                    chunk_strategy="fixed",
                )
                fout.write(json.dumps(asdict(chunk)) + "\n")
                total_chunks += 1

    duration = time.time() - t0
    size_mb = out_path.stat().st_size / (1024 * 1024)
    log.info("  %d chunks → %s (%.1f MB) in %.1fs",
             total_chunks, out_path.name, size_mb, duration)

    return {
        "repo": repo_name,
        "py_files": len(py_files),
        "strategy": "fixed",
        "total_chunks": total_chunks,
        "output_path": str(out_path),
        "output_mb": round(size_mb, 1),
        "duration_s": round(duration, 1),
    }

# Per-repo pipeline - Recursive chunking
def parse_recursive(repo_name: str, cfg: dict) -> dict:
    """
    Parse a repo using recursive-character chunking.
    Returns stats dict.
    """
    repo_dir = REPOS_DIR / repo_name
    if not repo_dir.exists():
        log.error("Repo not found: %s - run clone_repos.py first", repo_dir)
        return {}

    output_dir = Path("data/chunks_recursive")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{repo_name}.jsonl"

    py_files = [f for f in repo_dir.rglob("*.py") if not should_skip(f)]
    log.info("  %d .py files to parse (strategy=recursive)", len(py_files))

    total_chunks = 0
    t0 = time.time()
    chunking_cfg = cfg["chunking"]["recursive"]
    chunk_size = chunking_cfg.get("chunk_size", 512)
    chunk_overlap = chunking_cfg.get("chunk_overlap", 64)
    separators = chunking_cfg.get("separators", ["\nclass ", "\ndef ", "\n\n", "\n"])

    with out_path.open("w", encoding="utf-8") as fout:
        for filepath in py_files:
            try:
                text = filepath.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            repo_root = REPOS_DIR / repo_name
            rel_path = str(filepath.relative_to(repo_root))
            raw_chunks = chunk_recursive(text, chunk_size, chunk_overlap, separators)

            for line_start, line_end, chunk_text in raw_chunks:
                if not chunk_text.strip():
                    continue
                line_count = line_end - line_start + 1
                chunk_id = f"{repo_name}::{rel_path}::recursive::{line_start}"

                chunk = Chunk(
                    chunk_id=chunk_id,
                    repo=repo_name,
                    filepath=rel_path,
                    language="python",
                    function_name="",
                    class_name="",
                    is_method=False,
                    docstring="",
                    raw_code=chunk_text,
                    line_start=line_start,
                    line_end=line_end,
                    line_count=line_count,
                    chunk_strategy="recursive",
                )
                fout.write(json.dumps(asdict(chunk)) + "\n")
                total_chunks += 1

    duration = time.time() - t0
    size_mb = out_path.stat().st_size / (1024 * 1024)
    log.info("  %d chunks → %s (%.1f MB) in %.1fs",
             total_chunks, out_path.name, size_mb, duration)

    return {
        "repo": repo_name,
        "py_files": len(py_files),
        "strategy": "recursive",
        "total_chunks": total_chunks,
        "output_path": str(out_path),
        "output_mb": round(size_mb, 1),
        "duration_s": round(duration, 1),
    }


# Display utilities 
def print_summary(all_stats: list[dict], strategy: str) -> None:
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
    log.info("Chunks written to: data/chunks_%s/", strategy)
    log.info("")


def print_sample_chunks(repo_name: str, strategy: str, n: int = 3) -> None:
    """Print sample chunks for quality inspection."""
    out_path = Path(f"data/chunks_{strategy}") / f"{repo_name}.jsonl"
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
                "[%d] %s::%s  (lines %d-%d, %d loc)",
                i + 1,
                chunk["filepath"],
                chunk.get("function_name", "(code chunk)"),
                chunk["line_start"],
                chunk["line_end"],
                chunk["line_count"],
            )
            if chunk.get("docstring"):
                doc_preview = chunk["docstring"][:120].replace("\n", " ")
                log.info("     docstring: %s...", doc_preview)
            code_preview = chunk["raw_code"][:200].replace("\n", "↵ ")
            log.info("     code:      %s...", code_preview)
            log.info("")

# Entry point 
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified parser for all chunking strategies"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Parse a single repo (e.g. --repo numpy). Default: all repos.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print stats for already-parsed chunks without re-parsing.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Print sample chunks after parsing for quality inspection.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=["function", "fixed", "recursive"],
        help="Chunking strategy",
    )
    args = parser.parse_args()

    repos = [args.repo] if args.repo else REPO_NAMES
    cfg = load_config()
    chunk_dir = Path(f"data/chunks_{args.strategy}")

    # Stats-only mode
    if args.stats:
        all_stats = []
        for repo_name in repos:
            out_path = chunk_dir / f"{repo_name}.jsonl"
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
                "output_path": str(out_path),
                "duration_s": 0,
            })
        if all_stats:
            print_summary(all_stats, args.strategy)
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
    log.info("Substrate - Repository Parser")
    log.info("Strategy: %s", args.strategy)
    log.info("Repos   : %s", ", ".join(repos))
    log.info("Output  : %s/", chunk_dir)
    log.info("=" * 65)

    # Select parser based on strategy
    if args.strategy == "function":
        parse_func = lambda repo: parse_functions(repo, cfg, build_parser())
    elif args.strategy == "fixed":
        parse_func = lambda repo: parse_fixed(repo, cfg)
    else:  # recursive
        parse_func = lambda repo: parse_recursive(repo, cfg)

    all_stats = []
    for repo_name in repos:
        log.info("")
        log.info("Parsing %s...", repo_name)
        stats = parse_func(repo_name)
        if stats:
            all_stats.append(stats)
            if args.sample:
                print_sample_chunks(repo_name, args.strategy, n=2)

    if all_stats:
        print_summary(all_stats, args.strategy)


if __name__ == "__main__":
    main()
