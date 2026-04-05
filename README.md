---
title: Substrate
emoji: 🦀
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Substrate

> Cross-repo code reasoning via advanced RAG - function-boundary chunking,
> hybrid BM25+dense retrieval, cross-encoder re-ranking, and LLM-as-Judge evaluation.

**Demo query:**
```
"What functions in transformers would break if numpy changed the default
dtype of np.float_ from float64 to float32?"
```

## Setup (to run locally)

```bash
# 1. Clone this repo and switch to local configuration
git clone https://github.com/syedtaha22/substrate
cd substrate
git checkout local

# 2. Create virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env - add your HuggingFace API token
```

## Local LLM Setup (Required)

Install and start Ollama for local inference:

```bash
# Install Ollama: https://ollama.ai

# Pull required models
ollama pull llama3.1:8b
ollama pull mistral:7b

# Start Ollama server (runs at localhost:11434)
ollama serve
```

## Pipeline (run once, offline)

The pipeline clones 6 repositories, parses function boundaries, embeds chunks, and builds keyword indices.
Example shown for function-level chunks (A3, A5); repeat with `--strategy fixed` and `--strategy recursive` for other configurations.

```bash
# Step 1: Clone all 6 target repos (sparse, ~800MB)
python pipeline/clone_repos.py

# Step 2: Parse functions and extract chunks
python pipeline/parse_repos.py --strategy function
# Repeat for: --strategy fixed, --strategy recursive

# Step 3: Embed chunks and upsert to vector database
python pipeline/embed_and_upsert.py --strategy function

# Step 4: Build BM25 keyword index
python pipeline/build_bm25.py --strategy function
```

## Evaluation (Reproduce Paper Results)

All 33 evaluation queries are in `eval/test_queries.yaml`:

```bash
# Baseline: LLM with no retrieval context
python eval/eval_baseline.py
# Output: eval/results/baseline.json

# Retrieval-only evaluation (top-10 chunks, no generation)
python eval/eval_retrieval.py --strategy function

# Full RAG with LLM-as-Judge (faithfulness + relevancy scoring)
python eval/eval_rag.py --profile A5
# Takes ~2 hours on single GPU; run --profile A1 through A5 to test all configurations
```

Results JSON files contain per-query and per-tier metrics:
- `keyword_coverage`: % of expected keywords in answer
- `faithfulness`: % of answer claims grounded in retrieved code
- `answer_relevancy`: cosine similarity between query and generated-question embeddings
- `pass_rate`: % of queries exceeding threshold

## Running the App locally

The Chainlit app provides an interactive interface to test queries against the A5 profile.

```bash
# Start the app
chainlit run app/app.py --port 8000
# Open http://localhost:8000 in your browser
```