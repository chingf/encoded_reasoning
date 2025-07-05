import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from tqdm import tqdm
import json
import pickle
import gc
import os
import numpy as np
import torch.multiprocessing as mp
import time
import argparse
from config import storage_dir

def parse_args():
    parser = argparse.ArgumentParser(description="Generate transcripts using Llama-3.3-70B-Instruct model.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B", help="Name of the model.")
    parser.add_argument("--context_length", type=int, default=4096, help="Maximum context length for generation.")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index for dataset samples.")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum number of samples to process.")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for generation.")
    return parser.parse_args()


## HELPER FUNCTIONS

def format_conversation_for_qwen(user_prompt, tokenizer):
    formatted = tokenizer.apply_chat_template(
        user_prompt,
        tokenize=False,
        add_generation_prompt=True,
        #enable_thinking=False
    )
    return formatted

def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=False))

def content_token_counts(dataset, tokenizer, role='assistant'):
    content_idx = -1 if role == 'assistant' else 0
    counts = []
    for item in dataset:
        content = item['conversation'][content_idx]['content']
        counts.append(count_tokens(content, tokenizer))
    return np.array(counts)

def extract_and_count_tokens(response, tokenizer):
    # Extract content between <think> and </think>
    start_tag = "<think>"
    end_tag = "</think>"
    start_idx = response.find(start_tag)
    end_idx = response.find(end_tag)
    if start_idx == -1:
        raise ValueError("Start tag <think> not found in response.")
    if end_idx == -1:
        end_idx = len(response)
        finished_thinking = False
    else:
        finished_thinking = True
    
    extracted_content = response[start_idx + len(start_tag):end_idx]
    
    # Count tokens in the extracted content
    token_count = len(tokenizer.encode(extracted_content, add_special_tokens=False))
    
    return extracted_content, token_count, finished_thinking


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name
    context_length = args.context_length
    start_idx = args.start_idx
    max_samples = args.max_samples
    batch_size = args.batch_size

    model_storage_dir = os.path.join(storage_dir, "lm_sys", model_name.split("/")[-1])
    if not os.path.exists(model_storage_dir):
        os.makedirs(model_storage_dir)
    cache_dir = "/n/holylfs06/LABS/krajan_lab/Lab/cfang/hf_cache/"
    dataset = Dataset.load_from_disk(os.path.join(storage_dir, 'lm_sys', 'lm_sys_prompts_maxlen=200'))
    dataset = dataset['conversations']
    dataset = dataset[start_idx:start_idx + max_samples]
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        cache_dir=cache_dir, 
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    results = {'token_counts': [], 'finished_thinking': []}
    for i in range(0, len(dataset), batch_size):
        print(f'processing index {i} to {i + batch_size}')
        end_idx = min(i + batch_size, len(dataset))
        batch = dataset[i:end_idx]
        user_prompts = []
        user_turns = []
        for user_prompt in batch:
            formatted_prompt = format_conversation_for_qwen(user_prompt, tokenizer)
            user_prompts.append(formatted_prompt)
            user_turns.append(user_prompt)
        inputs = tokenizer(
            user_prompts,
            return_tensors="pt",
            padding_side='left',
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=context_length,
                do_sample=False,
            )
        for idx in range(len(user_prompts)):
            generated_ids = outputs[idx][inputs['input_ids'][idx].shape[0]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
            extracted_content, token_count, finished_thinking = extract_and_count_tokens(response, tokenizer)
            results['token_counts'].append(token_count)
            results['finished_thinking'].append(finished_thinking)
    
    with open('thinking_token_counts.py', 'wb') as f:
        pickle.dump(results, f)