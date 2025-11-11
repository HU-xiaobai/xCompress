# Code-of-Beyond-Perplexity-Let-the-Reader-Select-Retrieval-Summaries-via-Spectrum-Projection-Score

Paper Previous Name: Spectrum Projection Score: Aligning Retrieved Summaries with Reader Models in Retrieval-Augmented Generation

## Introduction

This repository accompanies our <span style="color:red">AAAI 2026 Oral🌟</span> paper:

**Beyond Perplexity: Let the Reader Select Retrieval Summaries via Spectrum Projection Score**  
Zhanghao Hu, Qinglin Zhu, Siya Qi, Yulan He, Hanqi Yan, Lin Gui  
_In The 40th Annual AAAI Conference on Artificial Intelligence (AAAI 2026), Vienna, Austria_

📄 [Paper on Huggingface](https://huggingface.co/papers/2508.05909) | 📄 [arXiv Preprint](https://arxiv.org/abs/2508.05909) 🔥[Project Page](https://zhanghao-aaai2026-sps.github.io/AAAI2026-SPS/)

> Large Language Models (LLMs) have shown improved generation performance through retrieval-augmented generation (RAG) following the retriever-reader paradigm, which supplements model inputs with externally retrieved knowledge. However, prior work often evaluates RAG holistically, assessing the retriever and reader jointly, making it difficult to isolate the true contribution of retrieval, particularly given the prompt sensitivity of LLMs used as readers. We move beyond perplexity and introduce Spectrum Projection Score (SPS), a lightweight, supervision-free metric that allows the reader to gauge the semantic alignment of a retrieved summary with its hidden representation by comparing the area formed by generated tokens from the summary, and the principal directions of subspace in the reader and to measure the relevance. Building on SPS we present xCompress, an inference-time controller framework that dynamically samples, ranks, and compresses retrieval summary candidates. Extensive experiments on five QA benchmarks with four open source LLMs show that SPS not only enhances performance across a range of tasks but also provides a principled perspective on the interaction between retrieval and generation.

---
