"""
app/app.py — Substrate
Run: chainlit run app/app.py --port 8000
"""

import sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import chainlit as cl

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

log.info("Substrate is ready")

# LLM call 
def call_llm(messages: list[dict], max_tokens: int = 512, temp: float = 0.1) -> tuple[str, float]:
    """Placeholder - returns constant message"""
    return (
        "## Substrate: Cross-Repository Code Reasoning\n\n"
        "A system for deep analysis of Python's scientific computing ecosystem.\n\n"
        "### Indexed Libraries\n\n"
        "| Library | Functions | Role |\n"
        "|---------|-----------|------|\n"
        "| numpy | 2,328 | Foundational numerical computing |\n"
        "| scipy | 6,752 | Scientific algorithms |\n"
        "| pandas | 6,292 | Data manipulation |\n"
        "| scikit-learn | 3,886 | Machine learning |\n"
        "| PyTorch | 33,968 | Deep learning |\n"
        "| Transformers | 28,450 | NLP models |\n"
        "| **Total** | **81,676** | |\n\n"
        "### Key Metrics\n\n"
        "- **Pass Rate (Hybrid Retrieval):** 100% at top-10 results\n"
        "- **Keyword Coverage:** 0.902 average\n"
        "- **Match Ranking Score:** 0.429\n"
        "- **Query Speed:** ~1s per query (BM25+dense with re-ranking)\n\n"
        "### Technical Approach\n\n"
        "- Function-level semantic chunking via tree-sitter\n"
        "- Hybrid BM25 + dense embedding retrieval\n"
        "- Cross-encoder re-ranking for result refinement\n"
        "- LLM-as-Judge evaluation framework\n\n"
        "**Status:** Minimal test deployment. Full reasoning capabilities and enhanced retrieval coming soon. Please check back later.",
        0.1
    )

# Retrieval routing - simplified
async def needs_retrieval(query: str, use_rag: bool) -> bool:
    return False

# Format context for LLM 
def build_context(chunks: list[dict]) -> str:
    return ""

# Settings 
@cl.on_chat_start
async def start():
    cl.user_session.set("settings", {"use_rag": False, "method": "hybrid", "top_k": 5})
    await cl.ChatSettings([
        cl.input_widget.Switch(
            id="use_rag", label="Enable RAG", initial=True,
            description="Retrieve source code before answering",
        ),
        cl.input_widget.Select(
            id="method", label="Retrieval method", initial_index=0,
            values=["hybrid", "dense", "bm25"],
            description="hybrid = BM25 + dense + RRF + reranking",
        ),
        cl.input_widget.Slider(
            id="top_k", label="Chunks to retrieve",
            initial=5, min=1, max=10, step=1,
        ),
    ]).send()

@cl.on_settings_update
async def update_settings(s: dict):
    cl.user_session.set("settings", {
        "use_rag": s.get("use_rag", True),
        "method":  s.get("method", "hybrid"),
        "top_k":   int(s.get("top_k", 5)),
    })

# Main handler 
@cl.on_message
async def on_message(msg: cl.Message):
    """Minimal message handler - returns constant message"""
    query = msg.content.strip()
    if not query:
        return

    # Generate constant response
    answer, _ = call_llm([], max_tokens=600, temp=0.1)
    
    # Send response
    await cl.Message(
        content=answer,
        elements=[],
    ).send()