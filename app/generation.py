"""
app/generation.py

LLM generation wrapper supporting two providers:
  - hf_router: HuggingFace router (cloud, free tier)
  - local:     Ollama (local, GPU recommended)

Provider is set in config.yaml -> generation.provider
Switch between them with one config change - no code changes needed.

Usage:
    from app.generation import Generator
    g = Generator()
    answer = g.generate("how does numpy clip work?", chunks=[])       # no RAG
    answer = g.generate("how does numpy clip work?", chunks=chunks)   # with RAG
"""

import json
import logging
import os
import time

import requests
import yaml

log = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# Prompts - loaded from config.yaml -> generation.system_prompt 
# Edit the prompt in config.yaml, not here.

RAG_TEMPLATE = """The following code was retrieved from the relevant repositories:

{context}

---

Question: {query}

Answer based on the code above:"""

NO_RAG_TEMPLATE = """Question: {query}

Answer:"""


class Generator:
    """
    LLM generation wrapper.
    Supports HF router (cloud) and Ollama (local) via config.yaml.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.cfg = load_config(config_path)
        
        gen_cfg = self.cfg["generation"]
        self.provider = gen_cfg.get("provider", "hf_router")
        self.model = gen_cfg["model"]
        self.max_new_tokens = gen_cfg.get("max_new_tokens", 512)
        self.temperature = gen_cfg.get("temperature", 0.1)
        self.hf_token = os.environ.get("HF_API_TOKEN", "")
        # Load system prompt from config (single source of truth)
        self.system_prompt = gen_cfg.get("system_prompt", "You are a helpful coding assistant.")

    # HF Router 
    def _call_hf_router(self, messages: list[dict], retries: int = 3, 
                        retry_delay: float = 10.0, ) -> str | None:
        
        url = "https://router.huggingface.co/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }

        for attempt in range(retries):
            try:
                resp = requests.post(
                    url, headers=headers,
                    data=json.dumps(payload), timeout=90
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"].strip()
                elif resp.status_code in (429, 529):
                    wait = retry_delay * (2 ** attempt)
                    log.warning("Rate limited - waiting %.0fs...", wait)
                    time.sleep(wait)
                elif resp.status_code == 503:
                    wait = retry_delay * (attempt + 1)
                    log.warning("Model loading - waiting %.0fs...", wait)
                    time.sleep(wait)
                else:
                    log.error("HF API error %d: %s", resp.status_code, resp.text[:300])
                    return None
            except requests.exceptions.Timeout:
                log.warning("Timeout (attempt %d/%d)", attempt + 1, retries)
                time.sleep(retry_delay)
            except Exception as e:
                log.error("Request error: %s", e)
                return None

        log.error("All %d attempts failed", retries)
        return None

    # Ollama (local) 

    def _call_ollama(self, messages: list[dict], retries: int = 3) -> str | None:
        """
        Call local Ollama server (http://localhost:11434).
        Install: curl -fsSL https://ollama.ai/install.sh | sh
        Pull model: ollama pull mistral  (or llama3.1, phi3, etc.)
        """
        url = "http://localhost:11434/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_new_tokens,
            },
        }

        for attempt in range(retries):
            try:
                resp = requests.post(url, json=payload, timeout=120)
                if resp.status_code == 200:
                    return resp.json()["message"]["content"].strip()
                else:
                    log.error("Ollama error %d: %s", resp.status_code, resp.text[:200])
                    return None
            except requests.exceptions.ConnectionError:
                log.error(
                    "Cannot connect to Ollama at localhost:11434. "
                    "Is it running? Start with: ollama serve"
                )
                return None
            except Exception as e:
                log.error("Ollama request error (attempt %d): %s", attempt + 1, e)
                time.sleep(2)

        return None

    # Public interface 
    def generate(self, query: str, chunks: list[dict] | None = None, 
                 context_str: str | None = None) -> dict:
        """
        Generate an answer for a query.

        Args:
            query:       The user's question
            chunks:      Retrieved chunks (optional - if None, no RAG)
            context_str: Pre-formatted context string (overrides chunks)

        Returns:
            dict with keys:
                answer:     Generated text
                has_rag:    Whether RAG context was used
                model:      Model name used
                provider:   Provider used
                duration_s: Time taken
        """
        t0 = time.time()

        # Build context
        if context_str:
            context = context_str
        elif chunks:
            from app.retrieval import Retriever
            r = Retriever.__new__(Retriever)
            r.cfg = self.cfg
            context = r.format_context(chunks)
        else:
            context = None

        has_rag = bool(context)

        # Build messages
        if has_rag:
            user_content = RAG_TEMPLATE.format(context=context, query=query)
        else:
            user_content = NO_RAG_TEMPLATE.format(query=query)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Call provider
        if self.provider == "local":
            answer = self._call_ollama(messages)
        else:
            answer = self._call_hf_router(messages)

        duration = time.time() - t0

        return {
            "answer": answer or "Error: no response from model.",
            "has_rag": has_rag,
            "model": self.model,
            "provider": self.provider,
            "duration_s": round(duration, 2),
        }