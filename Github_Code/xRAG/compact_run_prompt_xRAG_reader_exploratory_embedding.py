import os
import json
import numpy as np
import torch
import argparse
from tqdm import tqdm
import copy
import time
from sklearn.decomposition import PCA
from typing import Optional, List, Union, Tuple

from compact_api_config import CONFIG
from compact_evaluate import evaluate_QA

from compact_utils import get_logger, get_dataset, create_prompt, api_completion, parse_output_without_sentence,create_prompt_xRAG, parse_output_without_sentence_xRAG
from openai import OpenAI
import anthropic
import google.generativeai as genai


from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from src.model import SFR,XMistralForCausalLM
from src.language_modeling.utils import get_retrieval_embeds,XRAG_TOKEN

logger = get_logger(__name__)
def get_embedding_from_input_embeds(model,input_embeds: torch.Tensor, layer_index: int,
                                    attention_mask: Optional[torch.Tensor] = None):
    """
    使用 input_embeds 输入模型，提取指定层的 hidden states 并做 max pooling 得到句子表示。

    参数：
        input_embeds: Tensor of shape [1, seq_len, hidden_dim]
        layer_index: 指定要提取的 hidden state 层
        attention_mask: 可选的 attention mask，形状为 [1, seq_len]

    返回：
        Tensor: [hidden_dim] 的句子 embedding
    """
    if input_embeds.dim() == 2:
        input_embeds = input_embeds.unsqueeze(1)  # → [1, 1, hidden_dim]

    #print(f"input_embeds shape is {input_embeds.shape}")
    with torch.no_grad():
        outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=True)

    # 取指定层的 hidden state（[1, seq_len, hidden_dim]）
    hidden_states = outputs.hidden_states[layer_index].squeeze(0)  # shape: [seq_len, hidden_dim]

    # 如果提供了 attention mask，就屏蔽掉 padding 部分（用非常小的值）
    if attention_mask is not None:
        mask = attention_mask.squeeze(0).unsqueeze(-1)  # [seq_len, 1]
        hidden_states = hidden_states.masked_fill(mask == 0, float("-inf"))

    # max pooling 得到整体句子表示
    print(f"hidden_states shape is {hidden_states.shape}")
    sentence_embedding = hidden_states.max(dim=0).values  # shape: [hidden_dim]
    print(f"sentence_embedding shape is {sentence_embedding.shape}")
    # 如果是将Retrieval embedding只有一个embedding，那么max pooling, mean pooling, last token都没区别

    return sentence_embedding.cpu().numpy()

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
    if args.wo_prev_eval:
        args.checkpoint = f"{args.checkpoint}_wo_prev_eval"

    save_dir = os.path.join(args.compress_output_dir, args.compressor_name_or_path, args.checkpoint)
    logger.info(f"compress result save dir: {save_dir}")

    if os.path.isfile(os.path.join(save_dir, f'{args.results_file_name}.json')):
        logger.info("Already have results")
    
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # stop words
        stop = []
        #stop = list(set(stop + ["\n", "Ċ", "ĊĊ", "<0x0A>"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
        stop = list(set(stop + ["Ċ", "ĊĊ"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
        stop_token_ids = list(set([tokenizer.convert_tokens_to_ids(stop_token) for stop_token in stop] + [tokenizer.eos_token_id]))
        logger.info(f"no existing results compress ...")
        #ZH
        # —— 3. 从模型里取出“倒数第二层”的 query 矩阵 & 做 PCA ——
        #    假设模型 transformer block 在 model.model.layers
        layers = model.model.layers
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
                # —— 在这里截断到前 30 篇，但不动下面的 seg_idx 循环 ——
                for ae in active_examples:
                    docs = ae["example"]["documents_list"]
                    ae["example"]["documents_list"] = docs[:1] #ZH top 1 document select as the same in xRAG
                    #print(f"例子 {ae['index']} docs_list 截断后长度 = {len(ae['example']['documents_list'])}")

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

                        #ZH change input_prompt to document_input, make source document as the summary to it could
                        # be transferred into next
                        iteration['prompt'] = document_input #input_prompt
                        iteration['prompt_length'] = len(tokenizer(document_input).input_ids)
                        iteration['only_doc_prompt_length'] = len(tokenizer(document_input).input_ids)
                        iteration['summary'] = document_input #ZH make input prompt = summary to directly let LLM output
                        iteration['eval'] = " "
                        ae["iteration"] = iteration

                        inputs.append(document_input)

                    if not inputs:
                        continue
                    
                    tokenizer.padding_side = 'left'
                    tokenizer.pad_token = tokenizer.eos_token
                    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
                    model.config.pad_token_id = tokenizer.pad_token_id

                    model.resize_token_embeddings(len(tokenizer))
                    inputs_batch = tokenizer(inputs, return_tensors="pt", padding=True).to(device)

                    for ae_idx, ae in enumerate(active_examples):
                        iteration = ae["iteration"]
                        iteration['output'] = " "

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
                
                # Filter out completed examples only after all iterations are done
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
            json.dump(compress_results, open(os.path.join(save_dir, f'{args.results_file_name}.json'), 'w', encoding='utf-8'), indent=4)


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
                    # print(f"iteration {(i / segment_size) + 1}")
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

                    
                    # iteration['prev_input'] = prev_input
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

    
    
    
        logger.info(f"unload the compressor ... ")
        del model
        torch.cuda.empty_cache()

    
    
    """
    READ
    """
    # ZH xRAG model load
    retriever_name_or_path = "Salesforce/SFR-Embedding-Mistral"
    retriever = SFR.from_pretrained(retriever_name_or_path, torch_dtype=torch.bfloat16).eval().to(device)
    retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_name_or_path)
    if "xrag" in args.model_name_or_path:
        model = XMistralForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16,
                                                    low_cpu_mem_usage=True, ).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, add_eos_token=False, use_fast=False,
                                                  padding_side='left')

        ## here, XRAG_TOKEN is just a place holder
        model.set_xrag_token_id(tokenizer.convert_tokens_to_ids(XRAG_TOKEN))
        print(XRAG_TOKEN)

        layers = model.model.layers
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


    compress_path = os.path.join(args.compress_output_dir, args.compressor_name_or_path, args.checkpoint, f'{args.results_file_name}.json')
    comp = json.load(open(compress_path))

    compressed_context = {}
    sourced_document = {}

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
        #retrieval_original = "\n".join(d['iterations'][0]['documents_input_list'])
        #print(f"summary is {summary}")
        #print(f"d['question'] is {d['question']}")
        # ZH here summary is the source document! d is the example
        sourced_document[id] = summary
        #ZH source document become xRAG token here
        summary = create_prompt_xRAG(
            example=d,
            iteration=iteration,
            iter_idx=seg_idx,
            document_input=XRAG_TOKEN,
            prev_summary=prev_summary,
            prev_eval=prev_eval,
            tokenizer=tokenizer,
            eos_token="",
            add_generation_prompt=True,
        )

        if args.read_wo_prev_eval:
            compressed_context[id] = f"{summary}"
        elif args.read_wo_prev_summary:
            compressed_context[id] = f"{eval_reason}"
        #elif args.read_original_summary:
            #compressed_context[id] = f"{retrieval_original}"
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
        if "xrag" in args.model_name_or_path:
            print(XRAG_TOKEN)
            # 1) 准备要停的字符列表
            stop_chars = ["\n", "\\"]

            # 2) 对每个字符，先 tokenize 再 convert
            stop_token_ids = [tokenizer.eos_token_id]  # 保留原有的 eos
            for ch in stop_chars:
                toks = tokenizer.tokenize(ch)  # e.g. ['▁', '<0x0A>'] 或 ['▁\\']
                ids = tokenizer.convert_tokens_to_ids(toks)  # e.g. [28705, 13] 或 [414]
                print(f"{repr(ch)} -> tokens: {toks} -> ids: {ids}")
                stop_token_ids.extend(ids)

            # 3) 去重
            stop_token_ids = list(set(stop_token_ids))
            print("final stop_token_ids:", stop_token_ids)

            instruction = "Write a high-quality answer (under 10 words) for the given question using only the provided search results (some of which might be irrelevant)."

        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16, device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
            stop = []

            print(XRAG_TOKEN)
            if 'Llama-2' in args.model_name_or_path:
                stop = list(set(stop + ["\n", "Ċ", "ĊĊ", "<0x0A>"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
            elif 'Llama-3' in args.model_name_or_path:
                stop = list(set(stop + ["Ċ", "ĊĊ"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
            elif 'qwen' in args.model_name_or_path.lower():
                # Qwen 模型中，换行符使用标准的 "\n"
                stop = list(set(stop))
            else:
                raise AssertionError('No specified reader model')
            stop_token_ids = list(set([tokenizer.convert_tokens_to_ids(stop_token) for stop_token in stop] + [tokenizer.eos_token_id]))

            instruction = "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant)."

    print(f"stop_token_ids is {stop_token_ids}")

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

        question = f"Question: {d['question']}\nAnswer:" #ZH, here no for xRAGs
        
        if id in compressed_context:
            demonstration_str = compressed_context[id].strip('\n')
            sourced_document_str = sourced_document[id]
        else:
            print(id)
            # raise AssertionError("no compressed context")
            AssertionError("no compressed context")
            continue
            demonstration_str = ""
        
        
        prompt = "\n".join([instruction, demonstration_str])
        #print(f"prompt is {prompt}")
        #print(f"sourced_document_str is {sourced_document_str}")

        #ZH xRAG
        retriever_input = retriever_tokenizer(sourced_document_str, max_length=2000, padding=True, truncation=True,
                                              return_tensors='pt').to(device)
        with torch.no_grad():
            retrieval_sentence_embedding = retriever.get_doc_embedding(input_ids=retriever_input.input_ids,
                                                     attention_mask=retriever_input.attention_mask)
        #print(f"retrieval_sentence_embedding shape is {retrieval_sentence_embedding.shape}") torch.Size([1, 4096])

        # 1. 准备 prompt 的 input_ids 和 attention_mask
        tokenized = tokenizer(
            question,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length
        ).to(device)
        input_ids = tokenized.input_ids  # [1, seq_len]
        attention_mask = tokenized.attention_mask  # [1, seq_len]

        # 2. 把 input_ids → inputs_embeds
        with torch.no_grad():
            inputs_embeds = model.get_input_embeddings()(input_ids)  # [1, seq_len, hidden_size]

        # 3. 循环做探索
        num_runs = 50
        k = 6

        vars_and_expls = []  # list of (var, exploratory_embedding)

        for run in range(1, num_runs + 1):
            # 3.1 生成 exploratory embedding
            exploratory_embedding = torch.randn_like(retrieval_sentence_embedding)  # [1, hidden_size]

            # 3.2 拼到 inputs_embeds 后面，构造新的序列 embedding
            #     extended_inputs_embeds: [1, seq_len+1, hidden_size]
            extended_inputs_embeds = torch.cat(
                [inputs_embeds, exploratory_embedding.unsqueeze(1)],
                dim=1
            )

            # 3.3 对应地扩张 attention_mask
            #     extended_attention_mask: [1, seq_len+1]
            extra_mask = torch.ones((1, 1), dtype=attention_mask.dtype, device=device)
            extended_attention_mask = torch.cat([attention_mask, extra_mask], dim=1)

            # 3.4 前向，带上检索 embedding
            model = model.to(torch.float32)

            outputs = model(
                inputs_embeds=extended_inputs_embeds,
                attention_mask=extended_attention_mask,
                retrieval_embeds=retrieval_sentence_embedding.unsqueeze(1),  # [1,1,hidden]
                output_hidden_states=True,
                return_dict=True
            )

            # 3.5 取倒数第二层的 hidden states -> [1, seq_len+1, hidden] #ZH 这里层数需要和 pca error层数一样，记得改！change
            penult = outputs.hidden_states[-2]
            #      最后一个位置对应我们的 exploratory token -> [1, hidden]
            expl_vec = penult[:, -1, :]

            # 3.6 排序、top-k 差分、方差
            sorted_vals, _ = torch.sort(expl_vec, descending=True)  # [1, hidden]
            topk = sorted_vals[0, 1:k+1]  # [k]
            diffs = topk[:-1] - topk[1:]  # [k-1]
            var = diffs.var(unbiased=False).item()
            vars_and_expls.append((var, exploratory_embedding))

        vars_and_expls.sort(key=lambda x: x[0])
        #Zh select 10
        best_ten = [e for _, e in vars_and_expls[:10]]

        # 3) 构造 6 种 retrievals：原始 + 原始+10个 best exploratory
        retrieval_variants = [retrieval_sentence_embedding] + [
            retrieval_sentence_embedding + e for e in best_ten
        ]

        space_errors = []
        attention_mask_retrieval = torch.ones((1, 1), device=device, dtype=torch.long)
        rag_template = """[INST] Refer to the background document and answer the questions:

        Background: {document}

        Question: {question} [/INST] The answer is:"""

        for idx, ret in enumerate(retrieval_variants):
            # 3.1 先用 generate 拿 projector 后的 embedding
            prompt = rag_template.format_map(dict(question=question, document=XRAG_TOKEN))
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            _ = model.generate(
                input_ids=input_ids,
                do_sample=False,
                max_new_tokens=20,
                pad_token_id=tokenizer.pad_token_id,
                retrieval_embeds=ret.unsqueeze(1),
            )
            proj_emb = model.last_retrieval_embeds  # [1,1,hidden]

            # 3.2 z-score normalization
            mean = proj_emb.mean(dim=-1, keepdim=True)
            std = proj_emb.std(dim=-1, keepdim=True) + 1e-8
            norm_emb = (proj_emb - mean) / std  # [1,1,hidden]

            # 3.3 计算空间误差
            v = get_embedding_from_input_embeds(
                model,
                norm_emb,
                layer_index=-2, #ZH 这里层数需要和 pca error层数一样，记得改！change
                attention_mask=attention_mask_retrieval
            )  # 返回 numpy [hidden]
            v_pca = pca.transform(v.reshape(1, -1))
            v_rec = pca.inverse_transform(v_pca).flatten()
            err = np.linalg.norm(v - v_rec)
            space_errors.append(err)

        # 4) 打印
        '''
        print(f"Original (no exploratory): {space_errors[0]:.6f}")
        for i, err in enumerate(space_errors[1:], 1):
            print(f"With best exploratory #{i}: {err:.6f}")
        '''

        # space_errors[0] 是 original retrieval 的 error
        orig_err = space_errors[0]
        # 剩下的都是加了 exploratory 的 10 个
        expl_errors = space_errors[1:]

        # 找到哪一个最小
        min_err = min(expl_errors)
        min_idx = expl_errors.index(min_err) + 1  # +1 因为 space_errors[0] 是原始

        # 如果最优 exploratory 比原始还要好，就用那条；否则回退到原始
        if min_err < orig_err:
            #print(f"Selecting exploratory variant #{min_idx} (err={min_err:.6f} < orig={orig_err:.6f})")
            best_ret = retrieval_variants[min_idx]
        else:
            #print(f"Keeping original retrieval (orig={orig_err:.6f} <= best_expl={min_err:.6f})")
            best_ret = retrieval_sentence_embedding

        #print(f"best_ret shape is {best_ret.shape}")

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
                attention_mask = tokenizer(prompt, return_tensors="pt").attention_mask.to(device)
                start_time = time.time()
                outputs = model.generate(
                    input_ids=inputs,  # shape: (1, prompt_len)
                    attention_mask=attention_mask,  # 千万别忘了 mask！
                    do_sample=False,
                    max_new_tokens=10,
                    pad_token_id=tokenizer.pad_token_id,
                    retrieval_embeds=best_ret.unsqueeze(0),
                    eos_token_id=stop_token_ids,
                )

                end_time = time.time()


            time_taken = end_time - start_time
            total_read_time += time_taken
            result['time_taken'] = time_taken

            #ZH xRAG
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            result['generated_answers'] = decoded
            
            #result['generated_answers'] = tokenizer.decode(outputs[0][inputs.size(1):], skip_special_tokens=True).strip()
            #print(f"result['generated_answers'] is {result['generated_answers']}")
    
        if 'context' in result:
            result.pop('context')
        if 'ctxs' in result:
            result.pop('ctxs')
        read_results.append(result)
        
        if i % args.interval == args.interval - 1 or i == len(data) - 1:
            json.dump(read_results, open(os.path.join(save_dir, f'{args.results_file_name}.json'), 'w'), indent=4)

    logger.info(f"n_skip: {n_skip}")
    
    logger.info(f"total read time : {total_read_time}")
    metrics = evaluate_QA(read_results, ans_key='answers', predict_key='generated_answers')
    try:
        metrics['avg_comp_length'] = np.mean([result['only_doc_prompt_length'] for result in read_results])
    except:
        logger.info('no measured length')
    logger.info(f"metris: {metrics}")
    logger.info(f"{save_dir}")

    json.dump(read_results, open(os.path.join(save_dir, f'{args.read_file_name}.json'), 'w'), indent=4)
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