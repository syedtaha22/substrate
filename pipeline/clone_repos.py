"""
pipeline/clone_repos.py

Shallow-clones all 6 target repositories into data/repos/.
Uses sparse checkout to fetch only Python source files,
keeping total disk usage ~800MB instead of ~15GB.

Usage:
    python pipeline/clone_repos.py
    python pipeline/clone_repos.py --full   # full clone, no sparse checkout
"""

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Configuration 
REPOS = [
    {
        "name": "numpy",
        "url": "https://github.com/numpy/numpy.git",
        "description": "Bedrock - everything depends on it",
    },
    {
        "name": "scipy",
        "url": "https://github.com/scipy/scipy.git",
        "description": "Deep numpy coupling, scientific correctness",
    },
    {
        "name": "pandas",
        "url": "https://github.com/pandas-dev/pandas.git",
        "description": "Heavy numpy consumer, complex dtype logic",
    },
    {
        "name": "scikit-learn",
        "url": "https://github.com/scikit-learn/scikit-learn.git",
        "description": "numpy + scipy consumer, large API surface",
    },
    {
        "name": "pytorch",
        "url": "https://github.com/pytorch/pytorch.git",
        "description": "Gravity center of modern ML",
    },
    {
        "name": "transformers",
        "url": "https://github.com/huggingface/transformers.git",
        "description": "Top of the stack, touches everything",
    },
]

# Only fetch these file patterns (sparse checkout)
SPARSE_PATTERNS = [
    "*.py",
    "*.pyi",
    "*.pyx",   # Cython - numpy/scipy use this heavily
    "*.pxd",   # Cython headers
]

DATA_DIR = Path("data/repos")

# Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# Helpers 
@dataclass
class CloneResult:
    name: str
    success: bool
    path: Path
    duration_s: float
    error: str = ""

def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command, streaming output to the log."""
    log.debug("$ %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result

def clone_sparse(repo: dict, dest: Path) -> None:
    """
    Shallow sparse clone - only fetches Python/Cython source files.
    Keeps disk usage minimal while giving us everything we need to parse.
    """
    log.info("  Initialising sparse clone...")
    dest.mkdir(parents=True, exist_ok=True)

    run(["git", "init"], cwd=dest)
    run(["git", "remote", "add", "origin", repo["url"]], cwd=dest)

    # Enable sparse checkout
    run(["git", "sparse-checkout", "init", "--cone"], cwd=dest)

    # Write sparse-checkout patterns
    sparse_file = dest / ".git" / "info" / "sparse-checkout"
    sparse_file.write_text("\n".join(SPARSE_PATTERNS) + "\n")

    log.info("  Fetching (depth=1) - this may take a few minutes for large repos...")
    run(
        ["git", "fetch", "--depth=1", "origin", "HEAD"],
        cwd=dest,
        check=True,
    )
    run(
        ["git", "checkout", "FETCH_HEAD"],
        cwd=dest,
        check=True,
    )

def clone_full(repo: dict, dest: Path) -> None:
    """
    Full shallow clone - all files, depth=1.
    Use this if sparse checkout causes issues.
    """
    log.info("  Full shallow clone (depth=1)...")
    run(
        ["git", "clone", "--depth=1", repo["url"], str(dest)],
        check=True,
    )

def count_python_files(dest: Path) -> int:
    return sum(1 for _ in dest.rglob("*.py"))

def disk_usage_mb(dest: Path) -> float:
    result = run(["du", "-sm", str(dest)], check=False)
    try:
        return float(result.stdout.split()[0])
    except (IndexError, ValueError):
        return 0.0

# Main 
def clone_all(sparse: bool = True) -> list[CloneResult]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results: list[CloneResult] = []

    log.info("=" * 60)
    log.info("Substrate - Repository Cloning Pipeline")
    log.info("Mode    : %s", "sparse (Python/Cython only)" if sparse else "full")
    log.info("Target  : %s", DATA_DIR.resolve())
    log.info("Repos   : %d", len(REPOS))
    log.info("=" * 60)

    for i, repo in enumerate(REPOS, 1):
        dest = DATA_DIR / repo["name"]
        log.info("")
        log.info("[%d/%d] %s - %s", i, len(REPOS), repo["name"], repo["description"])

        # Skip if already cloned
        if (dest / ".git").exists():
            py_count = count_python_files(dest)
            log.info("  Already exists - %d .py files found. Skipping.", py_count)
            results.append(CloneResult(
                name=repo["name"], success=True, path=dest,
                duration_s=0.0
            ))
            continue

        t0 = time.time()
        try:
            if sparse:
                clone_sparse(repo, dest)
            else:
                clone_full(repo, dest)

            duration = time.time() - t0
            py_count = count_python_files(dest)
            mb = disk_usage_mb(dest)

            log.info("    Done in %.1fs - %d .py files - %.0f MB on disk",
                     duration, py_count, mb)

            results.append(CloneResult(
                name=repo["name"], success=True, path=dest,
                duration_s=duration
            ))

        except Exception as exc:
            duration = time.time() - t0
            log.error("    FAILED after %.1fs: %s", duration, exc)
            results.append(CloneResult(
                name=repo["name"], success=False, path=dest,
                duration_s=duration, error=str(exc)
            ))

    return results

def print_summary(results: list[CloneResult]) -> None:
    log.info("")
    log.info("=" * 60)
    log.info("Summary")
    log.info("=" * 60)

    total_py = 0
    total_mb = 0.0

    for r in results:
        status = " " if r.success else "✗"
        if r.success and r.path.exists():
            py = count_python_files(r.path)
            mb = disk_usage_mb(r.path)
            total_py += py
            total_mb += mb
            log.info("  %s  %-15s  %5d .py files  %6.0f MB  %.1fs", status, r.name, py, mb, r.duration_s)
        else:
            log.info("  %s  %-15s  FAILED: %s", status, r.name, r.error)

    log.info("")
    log.info("  Total: %d .py files across all repos", total_py)
    log.info("  Total: %.0f MB on disk", total_mb)

    failed = [r for r in results if not r.success]
    if failed:
        log.warning("")
        log.warning("  %d repo(s) failed. Re-run to retry.", len(failed))
        sys.exit(1)
    else:
        log.info("")
        log.info("  All repos cloned successfully.")
        log.info("  Next step: python pipeline/parse_functions.py --repo numpy")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clone target repositories for Substrate"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full clone instead of sparse (downloads all file types)",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Clone a single repo by name (e.g. --repo numpy)",
    )
    args = parser.parse_args()

    # Filter to single repo if requested
    global REPOS
    if args.repo:
        matching = [r for r in REPOS if r["name"] == args.repo]
        if not matching:
            valid = ", ".join(r["name"] for r in REPOS)
            log.error("Unknown repo '%s'. Valid names: %s", args.repo, valid)
            sys.exit(1)
        REPOS = matching

    results = clone_all(sparse=not args.full)
    print_summary(results)


if __name__ == "__main__":
    main()
