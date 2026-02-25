# xCompress

Paper Previous Name: Spectrum Projection Score: Aligning Retrieved Summaries with Reader Models in Retrieval-Augmented Generation

## Introduction

This repository accompanies our <span style="color:red">AAAI 2026 Oral🌟</span> paper:

**Beyond Perplexity: Let the Reader Select Retrieval Summaries via Spectrum Projection Score**  
Zhanghao Hu, Qinglin Zhu, Siya Qi, Yulan He, Hanqi Yan, Lin Gui  
_In The 40th Annual AAAI Conference on Artificial Intelligence (AAAI 2026), Vienna, Austria_

📄 [Paper on Huggingface](https://huggingface.co/papers/2508.05909) | 📄 [arXiv Preprint](https://arxiv.org/abs/2508.05909) 🔥[Project Page](https://zhanghao-aaai2026-sps.github.io/AAAI2026-SPS/)

> Large Language Models (LLMs) have shown improved generation performance through retrieval-augmented generation (RAG) following the retriever-reader paradigm, which supplements model inputs with externally retrieved knowledge. However, prior work often evaluates RAG holistically, assessing the retriever and reader jointly, making it difficult to isolate the true contribution of retrieval, particularly given the prompt sensitivity of LLMs used as readers. We move beyond perplexity and introduce Spectrum Projection Score (SPS), a lightweight, supervision-free metric that allows the reader to gauge the semantic alignment of a retrieved summary with its hidden representation by comparing the area formed by generated tokens from the summary, and the principal directions of subspace in the reader and to measure the relevance. Building on SPS we present xCompress, an inference-time controller framework that dynamically samples, ranks, and compresses retrieval summary candidates. Extensive experiments on five QA benchmarks with four open source LLMs show that SPS not only enhances performance across a range of tasks but also provides a principled perspective on the interaction between retrieval and generation.

---


## Datasets

We copy the CompAct dataset and please refer to the CompAct GitHub Code: https://github.com/dmis-lab/CompAct. And it could be placed in the ./data path.

## Quick Start

Run inference with a single command. Here, we provide some examples of our method with LLaMA 3.1 8b Ins Model in text-to-text pattern (CompAct) and Mistral Model in text-to-embedding pattern (xRAG).

Please pay attention that in our experiments we use an A100 80G GPU, and you might need to change the batch size to adapt to your GPUs.

Text-to-text pattern (CompAct):
```bash
CUDA_VISIBLE_DEVICES=1 python run_prompt_norm_filtering.py --task HotpotQA --data_path /xxx/xxx(your data path) --fshot --fshot_path /xxx/xxx(your fshot data path) --compress_output_dir /xxx/xxx(your compress output path) --read_output_dir /xxx/xxx(your reader output path) --compressor_name_or_path cwyoon99/CompAct-7b --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --cache_dir /xxx/xxx(your cache path) --batch_decoding --batch_size 10 --read_wo_prev_eval --segment_size 5 --max_iteration 6
```
You can try to change the value of the "TOP_PCT" parameter in the run_prompt_norm_filtering.py document to balance the cost and effectiveness. If you would like to move the filtering part and test all the query, you could try the run_prompt_all.py document, and an example is below:
```bash
CUDA_VISIBLE_DEVICES=1 python run_prompt_all.py --task HotpotQA --data_path /xxx/xxx(your data path) --fshot --fshot_path /xxx/xxx(your fshot data path) --compress_output_dir /xxx/xxx(your compress output path) --read_output_dir /xxx/xxx(your reader output path) --compressor_name_or_path cwyoon99/CompAct-7b --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --cache_dir /xxx/xxx(your cache path) --batch_decoding --batch_size 10 --read_wo_prev_eval --segment_size 5 --max_iteration 6
```

Text-to-embedding pattern (xRAG):
```bash
CUDA_VISIBLE_DEVICES=1 python compact_run_prompt_xRAG_reader_exploratory_embedding.py --task HotpotQA --data_path /xxx/xxx(your data path) --fshot --fshot_path /xxx/xxx(your fshot data path) --compress_output_dir /xxx/xxx(your compress output path) --read_output_dir /xxx/xxx(your reader output path) --compressor_name_or_path cwyoon99/CompAct-7b --model_name_or_path Hannibal046/xrag-7b --cache_dir /xxx/xxx(your cache path) --batch_decoding --batch_size 10 --read_wo_prev_eval --segment_size 1 --max_iteration 1
```

## Acknowledgements

This codebase is inspired by the [CompAct EMNLP 2024](https://github.com/dmis-lab/CompAct) and [xRAG NeurIPS 2024](https://github.com/Hannibal046/xRAG).
.
We thank the authors for making their code publicly available, which helped us design and implement several components of SPS and xCompress.

## Citation 
If you find this repository useful, please cite our paper:

```bash
@article{hu2025spectrum,
  title={Spectrum Projection Score: Aligning Retrieved Summaries with Reader Models in Retrieval-Augmented Generation},
  author={Hu, Zhanghao and Zhu, Qinglin and Qi, Siya and He, Yulan and Yan, Hanqi and Gui, Lin},
  journal={arXiv preprint arXiv:2508.05909},
  year={2025}
}
```

