# Substrate

## About

Substrate is a cross-repository code reasoning system designed for deep analysis of Python's scientific ecosystem. It indexes over 81,000 functions across numpy, scipy, pandas, scikit-learn, PyTorch, and Hugging Face transformers, enabling sophisticated queries about code architecture, dependencies, and hypothetical API changes.

## System Overview

The platform uses advanced indexing and retrieval techniques:

- **Function-level chunking**: Code parsed with tree-sitter for precise semantic boundaries
- **Hybrid retrieval**: BM25 keyword search combined with dense embeddings (all-MiniLM-L6-v2)
- **Intelligent re-ranking**: Cross-encoder models refine search results for relevance
- **Evaluation framework**: LLM-as-Judge methodology validates answer quality

## Indexed Libraries

The system tracks cross-repository dependencies across:

- numpy: 2,328 functions - The foundational numerical computing layer
- scipy: 6,752 functions - Scientific algorithms built on numpy
- pandas: 6,292 functions - Data manipulation leveraging numpy
- scikit-learn: 3,886 functions - Machine learning using numpy and scipy
- PyTorch: 33,968 functions - Deep learning framework
- Transformers: 28,450 functions - State-of-the-art NLP models

Total: 81,676 functions analyzed and indexed.

## Use Cases

This system helps you understand:

- Internal function implementations across major libraries
- Cross-library dependencies and integration patterns
- Impact of hypothetical API changes across dependent libraries
- How algorithms are implemented in the scientific Python stack

## Current Status

This is a test deployment. Enhanced functionality coming soon. Please check back later for the full platform features.
