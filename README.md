---
title: Substrate
emoji: 🦀
colorFrom: indigo
colorTo: blue
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

## Setup

```bash
# 1. Clone this repo
git clone https://github.com/your-handle/substrate
cd substrate

# 2. Create virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env - add your Pinecone API key and HF token
```

## Pipeline (run once, offline)

```bash
# Step 1: Clone all 6 target repos (sparse, ~800MB)
python pipeline/clone_repos.py

# Step 2: Parse functions with tree-sitter
python pipeline/parse_functions.py

# Step 3: Embed and upsert to Pinecone or ChromaDB
python pipeline/embed_and_upsert.py

# Step 4: Build BM25 index
python pipeline/build_bm25.py
```

## Run the app locally

```bash
chainlit run app/app.py --port 8000
```

## Deploy to Hugging Face Spaces

```bash
# Push the app/ directory to your HF Space
# See paper/substrate_paper.md for full deployment instructions
```

## Target Codebases

| Repo | Role |
|------|------|
| numpy/numpy | Bedrock |
| scipy/scipy | Deep numpy coupling |
| pandas-dev/pandas | numpy consumer |
| scikit-learn/scikit-learn | numpy + scipy consumer |
| pytorch/pytorch | ML gravity center |
| huggingface/transformers | Top of the stack |

## Research

See `paper/substrate_paper.md` for the full paper draft including
ablation study results and LLM-as-Judge evaluation methodology.
