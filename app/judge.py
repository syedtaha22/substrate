"""
app/judge.py

LLM-as-Judge for RAG quality evaluation using Ollama.

Metrics:
  - Faithfulness: Are answer claims grounded in retrieved code?
  - Relevancy: Does the answer address the question?
"""

import json
import logging
import re
from typing import Optional

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)


class LLMJudge:
    """Reusable LLM-as-Judge for evaluating RAG outputs."""

    # Prompts
    CLAIM_EXTRACT = """\
Extract every factual claim from this answer about Python code or libraries.
One claim per line. Be specific - include function names and behaviours.
Return ONLY a JSON array of strings.

Answer:
{answer}

JSON array of claims:"""

    CLAIM_VERIFY = """\
For each claim, answer YES if it is directly supported by the context, NO if not.
Return ONLY a JSON array of YES/NO in the same order as the claims.

Context (retrieved source code):
{context}

Claims:
{claims}

JSON array (YES/NO only):"""

    QUESTION_GEN = """\
Write {n} questions that the answer below would directly address.
Return ONLY a JSON array of question strings.

Answer:
{answer}

JSON array of {n} questions:"""

    def __init__(self, ollama_url: str = "http://localhost:11434", judge_model: str = "mistral"):
        """
        Initialize the LLM judge.

        Args:
            ollama_url: Ollama API endpoint
            judge_model: Model name for faithfulness/relevancy evaluation
        """
        self.ollama_url = ollama_url
        self.judge_model = judge_model

    def call_ollama(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        max_tokens: int = 512,
        temp: float = 0.0,
    ) -> Optional[str]:
        """
        Call Ollama API.

        Args:
            messages: Chat messages
            model: Model name (uses self.judge_model if not specified)
            max_tokens: Max output tokens
            temp: Temperature for generation

        Returns:
            Generated text or None on error
        """
        if model is None:
            model = self.judge_model

        try:
            resp = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temp, "num_predict": max_tokens},
                },
                timeout=120,
            )
            if resp.status_code == 200:
                return resp.json()["message"]["content"].strip()
            log.error("Ollama %d: %s", resp.status_code, resp.text[:150])
            return None
        except requests.exceptions.ConnectionError:
            log.error("Ollama not running. Start with: ollama serve")
            return None
        except Exception as e:
            log.error("Ollama: %s", e)
            return None

    @staticmethod
    def parse_string_list(raw: str) -> Optional[list[str]]:
        """
        Robustly parse a list of strings from LLM output.
        Handles: proper JSON, unquoted values, numbered lists, bare lines.
        """
        if not raw:
            return None

        # Strip markdown code fences
        clean = re.sub(r"```[a-z]*\n?", "", raw).strip()

        # Try standard JSON array first
        m = re.search(r"\[.*?\]", clean, re.DOTALL)
        if m:
            fragment = m.group(0)
            # Fix unquoted string values: [YES, NO] -> ["YES", "NO"]
            fragment = re.sub(
                r"\[([^\]\"]*)\]",
                lambda x: "["
                + ", ".join(
                    f'"{v.strip()}"'
                    for v in x.group(1).split(",")
                    if v.strip()
                )
                + "]",
                fragment,
            )
            try:
                result = json.loads(fragment)
                if isinstance(result, list) and result:
                    return [str(v) for v in result]
            except Exception:
                pass

        # Fallback: numbered list  "1. claim text" or "1) claim text"
        numbered = re.findall(r"^\s*\d+[.)]\s*(.+)", clean, re.MULTILINE)
        if numbered:
            return [s.strip() for s in numbered if s.strip()]

        # Fallback: one item per non-empty line
        lines = [
            l.strip()
            for l in clean.splitlines()
            if l.strip() and not l.strip().startswith("#")
        ]
        if lines:
            return lines

        return None

    @staticmethod
    def parse_yes_no_list(raw: str) -> Optional[list[str]]:
        """Parse a list of YES/NO answers from LLM output."""
        if not raw:
            return None

        clean = re.sub(r"```[a-z]*\n?", "", raw).strip().upper()

        # JSON array approach
        m = re.search(r"\[.*?\]", clean, re.DOTALL)
        if m:
            fragment = m.group(0)
            # Ensure values are quoted
            fragment = re.sub(r"\b(YES|NO)\b", r'"\1"', fragment)
            try:
                result = json.loads(fragment)
                if isinstance(result, list):
                    return [str(v).upper() for v in result]
            except Exception:
                pass

        # Fallback: one YES/NO per line
        lines = [l.strip() for l in clean.splitlines() if l.strip() in ("YES", "NO")]
        if lines:
            return lines

        # Fallback: count YES/NO anywhere in the text
        yeses = len(re.findall(r"\bYES\b", clean))
        nos = len(re.findall(r"\bNO\b", clean))
        if yeses + nos > 0:
            return ["YES"] * yeses + ["NO"] * nos

        return None

    def faithfulness(self, answer: str, context: str) -> tuple[Optional[float], dict]:
        """
        Evaluate faithfulness: verified_claims / total_claims

        Step 1: LLM extracts atomic claims from the answer
        Step 2: LLM verifies all claims at once against context
        Step 3: Compute ratio

        Args:
            answer: Generated answer text
            context: Retrieved code context

        Returns:
            (score, details_dict) where score is 0-1 or None on failure
        """
        if not answer or not context:
            return None, {}

        # Step 1 - extract claims
        raw1 = self.call_ollama(
            [{
                "role": "user",
                "content": self.CLAIM_EXTRACT.format(answer=answer[:2000]),
            }],
            max_tokens=400,
            temp=0.0,
        )
        claims = self.parse_string_list(raw1)
        if not claims:
            log.debug("  faith: could not extract claims from: %s", repr((raw1 or "")[:100]))
            return None, {"raw_claim_extract": raw1}
        claims = claims[:12]  # cap to keep runtime reasonable

        # Step 2 - batch verify
        numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
        raw2 = self.call_ollama(
            [{
                "role": "user",
                "content": self.CLAIM_VERIFY.format(context=context[:3000], claims=numbered),
            }],
            max_tokens=80,
            temp=0.0,
        )
        verdicts = self.parse_yes_no_list(raw2)
        if not verdicts:
            log.debug(
                "  faith: could not parse verdicts from: %s",
                repr((raw2 or "")[:100]),
            )
            return None, {"claims": claims, "raw_verdict_response": raw2}

        total = min(len(claims), len(verdicts))
        verified = sum(1 for v in verdicts[:total] if v == "YES")
        score = round(verified / total, 3)

        details = {
            "claims": claims[:total],
            "verdicts": verdicts[:total],
            "verified": verified,
            "total": total,
            "raw_claim_extract": raw1,
            "raw_verdict_response": raw2,
        }

        return score, details

    def relevancy(
        self,
        question: str,
        answer: str,
        embed_model: SentenceTransformer,
        n: int = 3,
    ) -> tuple[Optional[float], dict]:
        """
        Evaluate relevancy: mean cosine_similarity(generated_questions, original_question)

        Step 1: LLM generates N questions the answer addresses
        Step 2: Embed all + original query
        Step 3: Mean cosine similarity

        Args:
            question: Original user question
            answer: Generated answer text
            embed_model: SentenceTransformer model for embeddings
            n: Number of questions to generate

        Returns:
            (score, details_dict) where score is 0-1 or None on failure
        """
        if not answer or embed_model is None:
            return None, {}

        raw = self.call_ollama(
            [{
                "role": "user",
                "content": self.QUESTION_GEN.format(answer=answer[:2000], n=n),
            }],
            max_tokens=200,
            temp=0.3,
        )
        gen_qs = self.parse_string_list(raw)
        if not gen_qs:
            log.debug("  relev: could not parse questions from: %s", repr((raw or "")[:100]))
            return None, {"raw_question_generation": raw}
        gen_qs = gen_qs[:n]

        texts = [question] + gen_qs
        embs = embed_model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        orig = embs[0]
        sims = [float(np.dot(orig, e)) for e in embs[1:]]
        score = round(float(np.mean(sims)), 3)

        details = {
            "original_question": question,
            "generated_questions": gen_qs,
            "similarity_scores": sims,
            "mean_similarity": score,
            "raw_question_generation": raw,
        }

        return score, details
