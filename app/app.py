"""
app/app.py - Substrate
Run: chainlit run app/app.py --port 8000
"""

import sys, logging, json, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import chainlit as cl
import requests
import yaml
from dotenv import load_dotenv
load_dotenv()

from app.retrieval import Retriever
from app.generation import Generator

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Load once at startup 

log.info("Loading retrieval indexes...")
retriever = Retriever()
retriever.load()
generator = Generator()

with open("config.yaml") as f:
    _cfg = yaml.safe_load(f)

SYSTEM_PROMPT = _cfg["generation"]["system_prompt"].strip()
HF_TOKEN   = os.environ.get("HF_API_TOKEN", "")
MODEL      = generator.model
HF_URL     = "https://router.huggingface.co/v1/chat/completions"

log.info("Substrate ready - model: %s", MODEL)

# LLM call 
def call_llm(messages: list[dict], max_tokens: int = 512, temp: float = 0.1) -> tuple[str, float]:
    t0 = time.time()
    try:
        resp = requests.post(
            HF_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}",
                     "Content-Type": "application/json"},
            data=json.dumps({
                "model": MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temp,
            }),
            timeout=90,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip(), round(time.time()-t0, 1)
        return f"API error {resp.status_code}: {resp.text[:200]}", round(time.time()-t0, 1)
    except Exception as e:
        return f"Error: {e}", round(time.time()-t0, 1)

# Retrieval routing 
ROUTE_PROMPT = """You are a routing assistant. Decide if the user's message requires
searching a codebase index of numpy, scipy, pandas, scikit-learn, pytorch, and transformers.

Answer with exactly one word: YES or NO.

Search is needed for:
- Questions about how specific functions or classes work internally
- Questions about relationships between libraries
- Questions about implementation details, parameters, or source code
- Hypothetical questions about API changes and their effects
- Questions involving tracing a call chain or dependency

Search is NOT needed for:
- Greetings (hello, hi, how are you)
- Questions about Substrate itself or its capabilities
- General conversation
- Very broad conceptual questions with no specific library function in mind

User message: {query}
Answer (YES or NO):"""

OBVIOUS_NO = {
    "hello", "hi", "hey", "thanks", "thank you", "bye",
    "good morning", "good afternoon", "good evening",
}

async def needs_retrieval(query: str, use_rag: bool) -> bool:
    if not use_rag:
        return False
    q = query.lower().strip().rstrip("!?.")
    if q in OBVIOUS_NO or len(q.split()) <= 2:
        return False
    # Ask the model
    decision, _ = call_llm(
        [{"role": "user", "content": ROUTE_PROMPT.format(query=query)}],
        max_tokens=3, temp=0.0,
    )
    return "YES" in decision.upper()

# Format context for LLM 
def build_context(chunks: list[dict]) -> str:
    """
    Format chunks for the LLM. Each block is labelled filepath::functionname
    which is exactly what the LLM should cite in its answer.
    """
    parts = []
    for c in chunks:
        fp   = c.get("filepath", "?")
        fn   = c.get("function_name", "?")
        code = c.get("raw_code", c.get("_text", ""))[:700]
        label = f"{fp}::{fn}"
        parts.append(f"{label}\n```python\n{code}\n```")
    return "\n\n".join(parts)

# Settings 
@cl.on_chat_start
async def start():
    cl.user_session.set("settings", {"use_rag": True, "method": "hybrid", "top_k": 5})
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
    s       = cl.user_session.get("settings") or {}
    use_rag = s.get("use_rag", True)
    method  = s.get("method", "hybrid")
    top_k   = int(s.get("top_k", 5))
    query   = msg.content.strip()
    if not query:
        return

    chunks = []
    do_retrieve = await needs_retrieval(query, use_rag)

    # Retrieval 
    if do_retrieve:
        async with cl.Step(name="Searching codebase", type="retrieval") as step:
            chunks = retriever.retrieve(query, method=method, top_k=top_k)
            step.output = ""
            for c in chunks:
                repo   = c.get("repo", "?")
                fp     = c.get("filepath", "?")
                fn     = c.get("function_name", "?")
                line_s = c.get("line_start", "?")
                line_e = c.get("line_end", "?")
                step.output += f"**{fp}::{fn}** ({repo}, lines {line_s}-{line_e})\n"

    # Build messages 
    if chunks:
        user_content = (
            f"Retrieved source code:\n\n{build_context(chunks)}\n\n"
            f"---\n\nQuestion: {query}"
        )
    else:
        user_content = query

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    # Generate 
    async with cl.Step(name="Generating", type="llm") as step:
        answer, duration = call_llm(messages, max_tokens=600, temp=0.1)
        step.output = f"`{MODEL.split('/')[-1]}` - {duration}s"

    # Source elements 
    # Element names are "[1]", "[2]" etc - Chainlit renders these as clickable
    # links wherever the model writes [1], [2] in its answer text.
    elements: list[cl.Text] = []
    for c in chunks:
        repo   = c.get("repo", "?")
        fp     = c.get("filepath", "?")
        fn     = c.get("function_name", "?")
        score  = c.get("_rerank_score", c.get("_score", 0))
        line_s = c.get("line_start", "?")
        line_e = c.get("line_end", "?")
        code   = c.get("raw_code", c.get("_text", ""))

        elements.append(cl.Text(
            name=f"{fp}::{fn}",
            content=(
                f"### `{fn}()` - {repo}\n\n"
                f"**File:** `{fp}`  \n"
                f"**Lines:** {line_s}–{line_e}  ·  **Score:** `{score:.3f}`\n\n"
                f"```python\n{code}\n```"
            ),
            display="side",
        ))

    # Send 
    rag_info = f"{method} · {len(chunks)} chunks" if do_retrieve else "no retrieval"
    meta = f"\n\n---\n*{MODEL.split('/')[-1]} · {rag_info} · {duration}s*"

    await cl.Message(
        content=answer + meta,
        elements=elements,
    ).send()