import os
import json
import numpy as np
import torch
import argparse
from tqdm import tqdm
import copy
import time
from sklearn.decomposition import PCA
from api_config import CONFIG
from evaluate import evaluate_QA
from utils import get_logger, get_dataset, create_prompt, api_completion, parse_output_without_sentence
from openai import OpenAI
import anthropic
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

logger = get_logger(__name__)
def get_sentence_embedding(sentence: str, layer_index: int,model,tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # 取指定层 hidden_states，batch_size=1，形状 (num_tokens, hidden_dim)
    hs = outputs.hidden_states[layer_index].squeeze(0).cpu().numpy()
    # max pooling → (hidden_dim,)
    return hs.max(axis=0)

def get_sentence_embedding_mean(sentence: str, layer_index: int, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # 取指定层 hidden_states，batch_size=1，形状 (num_tokens, hidden_dim)
    hs = outputs.hidden_states[layer_index].squeeze(0).cpu().numpy()
    # mean pooling → (hidden_dim,)
    return hs.mean(axis=0)

# —— 4. 定义计算“空间误差”（PCA Reconstruction Error）的函数 ——
def compute_space_error(pca,model,tokenizer, sentence: str) -> float:
    """
    对一句话：
      1) 用 max-pooling 提取句向量 v 原始
      2) v 过 PCA → v_pca，再 inverse_transform 得到 v_rec
      3) 返回 ||v - v_rec||_2
    """
    v = get_sentence_embedding(sentence, layer_index=-2, model=model, tokenizer=tokenizer)
    v_pca = pca.transform(v.reshape(1, -1))              # (1, d_95)
    v_rec = pca.inverse_transform(v_pca).flatten()       # (hidden_dim,)
    return np.linalg.norm(v - v_rec)

# ===== 在文件顶部工具函数附近，新增一个计算比值的函数 =====
def compute_reader_ratio(summary_text: str, reader_model, reader_tokenizer, layer_index: int = -2) -> float:
    """
    用 reader 模型的倒数第二层 hidden_states 作为表示：
      - 对 seq_len 维做 mean pooling -> 向量 v_mean，取 L2 范数
      - 对 seq_len 维做 max  pooling -> 向量 v_max， 取 L1 范数
      - 返回 ratio = ||v_mean||_2 / (||v_max||_1 + 1e-12)
    """
    device = next(reader_model.parameters()).device
    with torch.no_grad():
        inp = reader_tokenizer(summary_text, return_tensors="pt", add_special_tokens=False).to(device)
        out = reader_model(**inp, output_hidden_states=True)
        hs = out.hidden_states[layer_index]  # shape: (1, seq_len, hidden_dim)
        # mean pooling over seq_len
        v_mean = hs.mean(dim=1).squeeze(0)      # (hidden_dim,)
        # max  pooling over seq_len
        v_max, _ = hs.max(dim=1)                # (1, hidden_dim)
        v_max = v_max.squeeze(0)                # (hidden_dim,)
        mean_l2 = torch.linalg.vector_norm(v_mean, ord=2).item()
        max_l1  = torch.linalg.vector_norm(v_max,  ord=1).item()
    return mean_l2 / (max_l1 + 1e-12)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not args.compressor_dir and not args.checkpoint:
        model_dir = args.compressor_name_or_path
    else:
        model_dir = os.path.join(args.compressor_dir, args.compressor_name_or_path, args.checkpoint)
    

    data = []
    with open(args.data_path, 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line))

    data_examples = get_dataset(data, n_docs=args.segment_size * args.max_iteration)
    print(f"data examples number is {len(data_examples)}")

    # data_examples=data_examples[911:912]
    # Add original index to each example
    for i, example in enumerate(data_examples):
        example["original_index"] = i


    """ 
    COMPRESS
    """
    compress_t0 = time.time()  # <<< 新增：压缩阶段起始时间
    if args.wo_prev_eval:
        args.checkpoint = f"{args.checkpoint}_wo_prev_eval"

    save_dir = os.path.join(args.compress_output_dir, args.compressor_name_or_path, args.checkpoint)
    logger.info(f"compress result save dir: {save_dir}")

    if os.path.isfile(os.path.join(save_dir, f'{args.results_file_name}.json')):
        logger.info("Already have results")
    
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        reader_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float32, device_map="auto") #原来是float16
        reader_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        # stop words
        stop = []
        #stop = list(set(stop + ["\n", "Ċ", "ĊĊ", "<0x0A>"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
        stop = list(set(stop + ["Ċ", "ĊĊ"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
        stop_token_ids = list(set([tokenizer.convert_tokens_to_ids(stop_token) for stop_token in stop] + [tokenizer.eos_token_id]))
        logger.info(f"no existing results compress ...")
        #ZH
        # —— 3. 从模型里取出“倒数第二层”的 query 矩阵 & 做 PCA ——
        #    假设模型 transformer block 在 model.model.layers
        #zhanghao
        layers = reader_model.model.layers
        #layers = model.model.layers
        n_layers = len(layers)
        target_layer = layers[n_layers - 2]

        # self_attn.q_proj.weight 的形状通常是 (hidden_size, hidden_size)
        q_weight = target_layer.self_attn.q_proj.weight.detach().cpu().numpy()

        # 3.1 全量 PCA，找出 95% 信息所需维度 d_95
        pca_full = PCA().fit(q_weight)
        cum_var = np.cumsum(pca_full.explained_variance_ratio_)
        d_95 = int(np.argmax(cum_var >= 0.95) + 1)

        # 3.2 用 d_95 做真正的降维 PCA
        pca = PCA(n_components=d_95).fit(q_weight)
        print(f"[PCA] 保留 95% 信息，降到 {d_95} 维")

        # ===== 在 main(args) 里，batch_decoding 分支开始之前，准备 ratio_pool =====
        # 在你已有的 reader_model / reader_tokenizer 加载完成之后（且在 batch_decoding 逻辑之前）加上：
        ### NEW: ratio pool config
        POOL_SIZE = 50
        TOP_PCT = 0.3  # top 30%
        ratio_pool = []  # 将在 batch 解码时持续维护
        examples_seen = 0  # 全局计数（跨 batch）

        if args.batch_decoding:
            """
            BATCH DECODING
            """
            
            tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")

            compress_results = []

            for idx in tqdm(range(0, len(data_examples), args.batch_size)):
                logger.info(f"batch {idx}")
                batch_examples = data_examples[idx:idx + args.batch_size]
                
                active_examples = [{"index": i, "example": ex, "iterations": [], "prev_summary": [], "prev_eval": []} for i, ex in enumerate(batch_examples)]
                
                for seg_idx in tqdm(range(0, max(len(ex['documents_list']) for ex in batch_examples), args.segment_size)):
                    if not active_examples:
                        break
                    
                    inputs = []
                    for ae in active_examples:
                        example = ae["example"]
                        documents_list = example['documents_list']
                        if seg_idx >= len(documents_list):
                            continue

                        iteration = {}
                        segment = documents_list[seg_idx:seg_idx + args.segment_size]
                        iteration['documents_input_list'] = [f"{doc['title']} {doc['text']}" for doc in segment]
                        document_input = "\n".join(iteration['documents_input_list'])

                        # split instruction version
                        if seg_idx == 0:
                            prev_summary = ""
                            prev_eval = ""
                        else:
                            try:
                                prev_summary = ae['prev_summary'][-1]
                                prev_eval = ae['prev_eval'][-1].replace('[INCOMPLETE]', '').strip()
                            except:
                                # import pdb; pdb.set_trace()
                                prev_summary = ""
                                prev_eval = ""

                        input_prompt = create_prompt(
                            example=example,
                            iteration=iteration,
                            iter_idx=seg_idx,
                            document_input=document_input,
                            prev_summary=prev_summary,
                            prev_eval=prev_eval,
                            tokenizer=tokenizer,
                            eos_token="",
                            add_generation_prompt=True,
                        )

                        #import pdb; pdb.set_trace()


                        iteration['prompt'] = input_prompt
                        iteration['prompt_length'] = len(tokenizer(input_prompt).input_ids)
                        iteration['only_doc_prompt_length'] = len(tokenizer(document_input).input_ids)
                        ae["iteration"] = iteration

                        inputs.append(input_prompt)

                    if not inputs:
                        continue
                    
                    tokenizer.padding_side = 'left'
                    tokenizer.pad_token = tokenizer.eos_token
                    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
                    model.config.pad_token_id = tokenizer.pad_token_id

                    model.resize_token_embeddings(len(tokenizer))
                    inputs_batch = tokenizer(inputs, return_tensors="pt", padding=True).to(device)

                    # ========= 第一步：对本 batch 的每个 prompt 先做一次“单次 deterministic summary” =========
                    ### NEW: 首先跑一次“单次生成”（不 sampling），得到每个 example 的 summary 文本
                    inputs_batch_once = tokenizer(inputs, return_tensors="pt", padding=True).to(device)
                    with torch.no_grad():
                        outputs_once = model.generate(
                            **inputs_batch_once,
                            max_new_tokens=900,
                            do_sample=False,
                            temperature=0,
                            top_p=1.0,
                            return_dict_in_generate=True
                        )
                    prompt_len_once = inputs_batch_once['input_ids'].shape[-1]

                    # 为每个 active example 暂存一次性 summary 和 ratio
                    batch_once_summaries = []
                    batch_once_ratios = []

                    for ae_idx, ae in enumerate(active_examples):
                        iteration = ae["iteration"]
                        once_text = tokenizer.decode(
                            outputs_once.sequences[ae_idx][prompt_len_once:],
                            skip_special_tokens=True
                        ).strip()
                        # 只取 summary 部分（Evaluation 前）
                        if "\n\nEvaluation:" in once_text:
                            once_summary = once_text.split("\n\nEvaluation:")[0]
                        else:
                            once_summary = once_text

                        batch_once_summaries.append(once_text)  # 完整文本（后续不采样时可直接用）
                        # 用 reader 的倒数第二层表示计算 ratio
                        ratio = compute_reader_ratio(once_summary, reader_model, reader_tokenizer, layer_index=-2)
                        batch_once_ratios.append(ratio)

                    # ========= 第二步：基于 ratio_pool 判定哪些需要 sampling，哪些直接用一次性结果 =========
                    ### NEW: 划分两个子集合
                    to_sample_indices = []
                    to_fast_indices = []

                    for local_idx, (ae, ratio) in enumerate(zip(active_examples, batch_once_ratios)):
                        examples_seen += 1

                        if len(ratio_pool) < POOL_SIZE:
                            # 前 50 个：无条件 sampling，并把比值放进池
                            to_sample_indices.append(local_idx)
                            ratio_pool.append(ratio)
                        else:
                            # 计算 30% 分位阈值（选出前 30% 进 sampling）
                            sorted_pool = sorted(ratio_pool)  # 升序
                            k = int(TOP_PCT * len(sorted_pool)) - 1
                            k = max(0, min(len(sorted_pool) - 1, k))
                            threshold = sorted_pool[k]

                            if ratio <= threshold:
                                # 命中 bottom 30%：需要 sampling
                                to_sample_indices.append(local_idx)
                                min_pos = ratio_pool.index(min(ratio_pool))  # 删最小
                                ratio_pool[min_pos] = ratio
                            else:
                                # 不 sampling
                                to_fast_indices.append(local_idx)
                                max_pos = ratio_pool.index(max(ratio_pool))  # 删最大
                                ratio_pool[max_pos] = ratio
                    # ========= 第三步：对“不 sampling”的那部分，直接用刚才的一次性结果入账 =========
                    ### NEW: 快速路径（沿用你给的“不需要 summary sampling 的参考代码”）
                    for idx_in_batch in to_fast_indices:
                        ae = active_examples[idx_in_batch]
                        iteration = ae["iteration"]
                        once_text = batch_once_summaries[idx_in_batch]

                        iteration['output'] = once_text
                        try:
                            parsed_sections = parse_output_without_sentence(once_text)
                        except Exception as e:
                            print(f"ERROR (fast path) parsing example {idx_in_batch}: {e}")
                            continue
                        iteration.update(parsed_sections)
                        ae["iterations"].append(iteration)
                        ae["prev_summary"].append(iteration.get('summary', ''))
                        ae["prev_eval"].append(iteration.get('eval', ''))

                        if "[COMPLETE]" in iteration.get('eval', ''):
                            ae["complete"] = True
                            result = copy.deepcopy(ae["example"])
                            result.pop('documents_list', None)
                            result.pop('documents', None)
                            result['iterations'] = ae["iterations"]
                            result['prev_summary'] = ae["prev_summary"]
                            result['prev_eval'] = ae["prev_eval"]
                            compress_results.append(result)

                    # 如果全是快速路径，就可以直接进入本轮末尾的收尾逻辑
                    if len(to_sample_indices) == 0:
                        active_examples = [ae for ae in active_examples if not ae.get("complete")]
                        continue  # 进入下一轮 seg_idx

                    # ========= 第四步：对“需要 sampling”的那部分，运行 5-beam + 你的 SPS 选择 =========
                    ### NEW: 仅对 to_sample_indices 二次生成（5-beam）
                    sample_inputs = [inputs[i] for i in to_sample_indices]
                    inputs_batch_sample = tokenizer(sample_inputs, return_tensors="pt", padding=True).to(device)

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs_batch_sample,
                            max_new_tokens=900,
                            do_sample=False,
                            num_beams=5,
                            num_return_sequences=5,
                            return_dict_in_generate=True,
                            output_scores=True,
                            repetition_penalty=1.2
                        )

                    prompt_len = inputs_batch_sample['input_ids'].shape[-1]
                    num_beams = 5

                    for j, idx_in_batch in enumerate(to_sample_indices):
                        ae = active_examples[idx_in_batch]
                        iteration = ae["iteration"]

                        start = j * num_beams
                        end = start + num_beams
                        seqs = outputs.sequences[start:end]
                        full_texts = [
                            tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
                            for seq in seqs
                        ]

                        summaries = []
                        for txt in full_texts:
                            if "\n\nEvaluation:" in txt:
                                summary_part = txt.split("\n\nEvaluation:")[0]
                            else:
                                summary_part = txt
                            summaries.append(summary_part)

                        # 仍然用你已有的 compute_space_error + PCA 选择最小误差的 beam
                        errors = [compute_space_error(pca, reader_model, reader_tokenizer, summary) for summary in
                                  summaries]
                        best_idx = int(np.argmin(errors))
                        best_full = full_texts[best_idx]
                        best_sum = summaries[best_idx]
                        best_err = errors[best_idx]
                        best_score = outputs.sequences_scores[start:end][best_idx].item() if hasattr(outputs,
                                                                                                     "sequences_scores") else None

                        iteration['output'] = best_full
                        iteration['space_error'] = best_err
                        iteration['beam_score'] = best_score

                        try:
                            parsed_sections = parse_output_without_sentence(best_full)
                        except Exception as e:
                            print(f"ERROR parsing best beam for example {idx_in_batch}: {e}")
                            continue

                        iteration.update(parsed_sections)
                        iteration['summary'] = iteration.get('summary', best_sum)
                        iteration['eval'] = iteration.get('eval', '[INCOMPLETE]')

                        ae["iterations"].append(iteration)
                        ae["prev_summary"].append(iteration['summary'])
                        ae["prev_eval"].append(iteration['eval'])

                        if "[COMPLETE]" in iteration['eval']:
                            ae["complete"] = True
                            result = copy.deepcopy(ae["example"])
                            result.pop('documents_list', None)
                            result.pop('documents', None)
                            result['iterations'] = ae["iterations"]
                            result['prev_summary'] = ae["prev_summary"]
                            result['prev_eval'] = ae["prev_eval"]
                            compress_results.append(result)

                    # 本轮末尾，和你原逻辑一致：去掉已完成的
                    active_examples = [ae for ae in active_examples if not ae.get("complete")]

                for ae in active_examples:
                    result = copy.deepcopy(ae["example"])
                    result.pop('documents_list', None)
                    result.pop('documents', None)
                    result['iterations'] = ae["iterations"]
                    result['prev_summary'] = ae["prev_summary"]
                    result['prev_eval'] = ae["prev_eval"]
                    compress_results.append(result)

            compress_results = sorted(compress_results, key=lambda x: x["original_index"])

            for result in compress_results:
                result.pop("original_index", None)

                
            os.makedirs(save_dir, exist_ok=True)

            def to_serializable(obj):
                if isinstance(obj, np.generic):
                    return obj.item()  # 转成 Python float/int
                raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

            json.dump(compress_results,
                      open(os.path.join(save_dir, f'{args.results_file_name}.json'), 'w', encoding='utf-8'),
                      indent=4,
                      default=to_serializable)


        else:
            # raise AssertionError("prevent single decoding")
            compress_results = []
            total_compress_time = 0
            for idx, example in enumerate(tqdm(data_examples[:])):
                documents_list = example['documents_list']

                iterations = []
                prev_summary = []
                prev_eval = []

                for i in tqdm(range(0, len(example['documents_list']), args.segment_size)):
                    iteration = {}
                    
                    segment = documents_list[i:i + args.segment_size]
                    iteration['documents_input_list'] = [f"{doc['title']} {doc['text']}" for doc in segment]
                    document_input = "\n".join(iteration['documents_input_list'])

                    # split instruction version
                    if i == 0:
                        prev_summary_temp = ""
                        prev_eval_temp = ""
                    else:
                        prev_summary_temp = prev_summary[-1]
                        prev_eval_temp = prev_eval[-1].replace('[INCOMPLETE]', '').strip()

                    input_prompt = create_prompt(
                        example=example,
                        iteration=iteration,
                        iter_idx=i,
                        document_input=document_input,
                        prev_summary=prev_summary_temp,
                        prev_eval=prev_eval_temp,
                        tokenizer=tokenizer,
                        eos_token="",
                        add_generation_prompt=True,
                        )


                    iteration['prompt'] = input_prompt
                    iteration['prompt_length']= len(tokenizer(input_prompt).input_ids)
                    iteration['only_doc_prompt_length'] = len(tokenizer(document_input).input_ids)
                    
                    with torch.no_grad():
                        inputs = tokenizer(input_prompt, return_tensors="pt")
                        input_ids = inputs.input_ids.to(device)
                        attention_mask = inputs.attention_mask.to(device)
                        start_time = time.time()
                        outputs = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=900,
                            do_sample=False,
                            temperature=0,
                            top_p=1.0,
                            pad_token_id=tokenizer.eos_token_id,
                            )
                        end_time = time.time()
                    iteration['output'] = tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
                    
                    time_taken = end_time - start_time
                    iteration['time_taken'] = time_taken
                    total_compress_time += time_taken

                    try:
                        parsed_sections = parse_output_without_sentence(iteration['output'])
                    except Exception as e:
                        print(f"ERROR: {e}")
                        iterations.append(iteration)
                        break

                    iteration.update(parsed_sections)
                    
                    iterations.append(iteration)
                    prev_summary.append(iteration['summary'])
                    prev_eval.append(iteration['eval'])

                    if "[COMPLETE]" in iteration['eval']:
                        break

                result = copy.deepcopy(example)
                result.pop('documents_list', None)
                result.pop('documents', None)
                
                result['iterations'] = iterations
                result['prev_summary'] = prev_summary
                result['prev_eval'] = prev_eval

                compress_results.append(result)

                os.makedirs(save_dir, exist_ok=True)
                if idx % args.interval == args.interval - 1 or idx == len(data_examples) - 1:
                    json.dump(compress_results, open(os.path.join(save_dir, f'{args.results_file_name}.json'), 'w', encoding='utf-8'), indent=4)
            
            logger.info(f"total compression time: {total_compress_time}")

        # <<< 新增：将 compress 耗时写到同目录
        compress_elapsed = time.time() - compress_t0
        time_dir = os.path.join(save_dir, "compress")
        os.makedirs(time_dir, exist_ok=True)
        with open(os.path.join(time_dir, "time.txt"), "w") as f:
            f.write(f"{compress_elapsed:.3f}\n")

        logger.info(f"unload the compressor ... ")
        del model
        torch.cuda.empty_cache()

    
    
    """
    READ
    """

    compress_path = os.path.join(args.compress_output_dir, args.compressor_name_or_path, args.checkpoint, f'{args.results_file_name}.json')
    comp = json.load(open(compress_path))

    compressed_context = {}
    for d in comp:
        if '_id' in d:
            id = d['_id']
        else:
            if 'id' in d:
                id = d['id']
            else:
                id = d['question']

        if len(d["prev_summary"]) <= args.max_iteration:
            try:
                summary = d["prev_summary"][-1]
                eval_reason = d['prev_eval'][-1]
            except:
                summary = ""
                eval_reason = ""
            # summary = d["prev_summary"][-1]
            # eval_reason = d['prev_eval'][-1]
        elif len(d["prev_summary"]) > args.max_iteration:
            summary = d["prev_summary"][args.max_iteration - 1]
            eval_reason = d['prev_eval'][args.max_iteration - 1]
            
        eval_reason = eval_reason.replace('[INCOMPLETE]','').replace('[COMPLETE]','')
        eval_reason = eval_reason.replace('\n','').strip()
        retrieval_original = "\n".join(d['iterations'][0]['documents_input_list'])

        if args.read_wo_prev_eval:
            compressed_context[id] = f"{summary}"
        elif args.read_wo_prev_summary:
            compressed_context[id] = f"{eval_reason}"
        elif args.read_original_summary:
            compressed_context[id] = f"{retrieval_original}"
        else:
            compressed_context[id] = f"{summary} {eval_reason}"

    save_dir = os.path.join(args.read_output_dir, args.compressor_name_or_path, args.checkpoint, args.model_name_or_path)
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"read result save dir: {save_dir}")

    


    logger.info(f"READER: {args.model_name_or_path}")

    if 'gpt' in args.model_name_or_path:
        api_key = CONFIG['openai_key'][0]
        client = OpenAI(api_key=api_key)
        
        # Due to overly verbose tendency of gpts, we add a short guideline (high-quality short answer (under 10 words))
        instruction = "Write a high-quality short answer (under 10 words) for the given question using the provided search results (some of which might be irrelevant)."
    elif 'claude' in args.model_name_or_path:
        api_key = CONFIG['anthropic_key'][0]
        client = anthropic.Anthropic(api_key=api_key)

        instruction = "Write a high-quality short answer (under 10 words) for the given question using the provided search results (some of which might be irrelevant). Follow the answer format of examples."
    elif 'gemini' in args.model_name_or_path:
        api_key = CONFIG['google_key'][0]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(args.model_name_or_path)
        
        instruction = "Write a high-quality short answer (under 10 words) for the given question using the provided search results (some of which might be irrelevant)."
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        stop = []
        if 'Llama-2' in args.model_name_or_path:
            stop = list(set(stop + ["\n", "Ċ", "ĊĊ", "<0x0A>"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
        elif 'Llama-3' in args.model_name_or_path:
            stop = list(set(stop + ["Ċ", "ĊĊ"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
        else:
            raise AssertionError('No specified reader model')
        stop_token_ids = list(set([tokenizer.convert_tokens_to_ids(stop_token) for stop_token in stop] + [tokenizer.eos_token_id]))

        instruction = "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant)."
    
    if args.fshot:
        fshot = json.load(open(args.fshot_path))

        if fshot:
            fixed_examples = [f"Question: {fs['question']}\nAnswer: {fs['answers'][0]}" for fs in fshot]
            fixed_examples="\n\n".join(fixed_examples)+"\n"

        instruction += f"\n\n{fixed_examples}"

        
    read_results = []
    total_read_time = 0
    n_skip = 0 # Some instances are rejected to answer by proprietary models.
    for i, d in enumerate(tqdm(data[:])):
        if '_id' in d:
            id = d['_id']
        else:
            if 'id' in d:
                id = d['id']
            else:
                id = d['question']

        question = f"Question: {d['question']}\nAnswer:"
        
        if id in compressed_context:
            demonstration_str = compressed_context[id].strip('\n')
        else:
            print(id)
            # raise AssertionError("no compressed context")
            AssertionError("no compressed context")
            continue
            demonstration_str = ""
        
        
        prompt = "\n".join([instruction, demonstration_str, question])
        
        result = copy.deepcopy(d)            
        result['prompt'] = prompt
        result['demonstration'] = demonstration_str

        if 'gpt' in args.model_name_or_path:
            response = api_completion(prompt, client, args.model_name_or_path, max_tokens=args.generation_max_length)
            result['usage'] = dict(dict(response).get('usage'))
            result['generated_answers'] = dict(dict(dict(response)['choices'][0])['message'])['content']
        elif 'claude' in args.model_name_or_path:
            response = api_completion(prompt, client, args.model_name_or_path, max_tokens=args.generation_max_length)
            result['generated_answers'] = response.content[0].text
        elif 'gemini' in args.model_name_or_path:
            response = model.generate_content(prompt)
            try:
                result['generated_answers'] = response.text
            except Exception as e:
                n_skip += 1
                print(e)
                continue
        else:
            result['prompt_length']= len(tokenizer(prompt).input_ids)
            result['only_doc_prompt_length'] = len(tokenizer(demonstration_str).input_ids)
            
            with torch.no_grad():
                inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                start_time = time.time()
                outputs = model.generate(
                    inputs,
                    max_new_tokens=args.generation_max_length,
                    do_sample=False,
                    temperature=0,
                    top_p=1.0,
                    eos_token_id=stop_token_ids,
                    )
                end_time = time.time()


            time_taken = end_time - start_time
            total_read_time += time_taken
            result['time_taken'] = time_taken
            
            result['generated_answers'] = tokenizer.decode(outputs[0][inputs.size(1):], skip_special_tokens=True).strip()
    
        if 'context' in result:
            result.pop('context')
        if 'ctxs' in result:
            result.pop('ctxs')
        read_results.append(result)
        
        if i % args.interval == args.interval - 1 or i == len(data) - 1:

            def to_serializable(obj):
                if isinstance(obj, np.generic):
                    return obj.item()  # 转成 Python float/int
                raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

            json.dump(read_results,
                      open(os.path.join(save_dir, f'{args.results_file_name}.json'), 'w', encoding='utf-8'),
                      indent=4,
                      default=to_serializable)
            #json.dump(read_results, open(os.path.join(save_dir, f'{args.results_file_name}.json'), 'w'), indent=4)

    logger.info(f"n_skip: {n_skip}")
    
    logger.info(f"total read time : {total_read_time}")
    metrics = evaluate_QA(read_results, ans_key='answers', predict_key='generated_answers')
    try:
        metrics['avg_comp_length'] = np.mean([result['only_doc_prompt_length'] for result in read_results])
    except:
        logger.info('no measured length')
    logger.info(f"metris: {metrics}")
    logger.info(f"{save_dir}")

    def to_serializable(obj):
        if isinstance(obj, np.generic):
            return obj.item()  # 转成 Python float/int
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    json.dump(read_results,
              open(os.path.join(save_dir, f'{args.results_file_name}.json'), 'w', encoding='utf-8'),
              indent=4,
              default=to_serializable)
    with open((os.path.join(save_dir, f'{args.metrics_file_name}.txt')),'w') as f:
        f.write(json.dumps(metrics))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--fshot_path', type=str)
    parser.add_argument('--segment_size', type=int, default=5)
    parser.add_argument('--max_iteration', type=int, default=6)

    parser.add_argument('--batch_decoding', action="store_true", default=False)
    parser.add_argument('--batch_size', type=int, default=100)

    # compress
    parser.add_argument('--compressor_name_or_path', type=str)
    parser.add_argument('--compressor_dir', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--compress_output_dir', type=str, required=True)
    parser.add_argument('--read_output_dir', type=str, required=True)
    parser.add_argument('--wo_prev_eval', action="store_true", default=False)
    parser.add_argument('--results_file_name', type=str, default='results')
    parser.add_argument('--read_file_name', type=str, default='results')
    parser.add_argument('--metrics_file_name', type=str, default='metrics')

    # read
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--cache_dir', type=str)
    parser.add_argument('--interval', type=int, default=2)
    parser.add_argument('--fshot', action='store_true', default=False)
    parser.add_argument('--read_wo_prev_eval', action="store_true", default=False)
    parser.add_argument('--read_wo_prev_summary', action="store_true", default=False)
    parser.add_argument('--read_original_summary', action="store_true", default=False)
    parser.add_argument("--do_sample", action="store_true", help="whether to use sampling (false is greedy)")
    parser.add_argument("--generation_max_length", type=int, default=32, help="max number of tokens to generate")
    parser.add_argument("--generation_min_length", type=int, default=0, help="min number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0, help="generation temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="top-p parameter for nucleus sampling")
    parser.add_argument("--no_cuda", action="store_true", help="disable cuda")
    parser.add_argument("--no_bf16", action="store_true", help="disable bfloat16 -- use fp32 instead")
    parser.add_argument("--debug", action="store_true", help="for debugging")
    
    
    # wandb
    parser.add_argument(
        "--use_wandb", action="store_true", default=False, help=""
    )
    parser.add_argument(
        "--wandb_project_name", type=str, default='', help=""
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default='test', help=""
    )
    
    args = parser.parse_args()

    main(args)