import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from tqdm import tqdm
import json
import gc
import os
import torch.multiprocessing as mp
import time
import argparse
from config import storage_dir

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Generate transcripts using Llama-3.3-70B-Instruct model.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B", help="Name of the model.")
    parser.add_argument("--context_length", type=int, default=400, help="Maximum context length for generation.")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index for dataset samples.")
    parser.add_argument("--max_samples", type=int, default=10000, help="Maximum number of samples to process.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for generation.")
    return parser.parse_args()


def format_conversation_for_qwen(user_prompt, tokenizer):
    formatted = tokenizer.apply_chat_template(
        user_prompt,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    return formatted

if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name
    context_length = args.context_length
    start_idx = args.start_idx
    max_samples = args.max_samples
    batch_size = args.batch_size

    # Directory handling and loading data
    start = time.time()
    model_storage_dir = os.path.join(storage_dir, "lm_sys", model_name.split("/")[-1])
    if not os.path.exists(model_storage_dir):
        os.makedirs(model_storage_dir)
    cache_dir = "/n/holylfs06/LABS/krajan_lab/Lab/cfang/hf_cache/"
    response_dir = os.path.join(model_storage_dir, f'lm_sys_{start_idx}_{start_idx+max_samples}')
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

    # Actually run processing
    processed_data = []
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
            max_length=context_length
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=context_length,
                do_sample=False,
            )
        for idx in range(len(user_prompts)):
            input_len = inputs['input_ids'][idx].shape[0]
            generated_ids = outputs[idx][inputs['input_ids'][idx].shape[0]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            new_conversation = [user_turns[idx][0], {'role': 'assistant', 'content': response}]
            processed_data.append({'conversation': new_conversation})

    # Save new dataset
    new_dataset = Dataset.from_list(processed_data)
    print(f"Created dataset with {len(new_dataset)} conversations")
    new_dataset.save_to_disk(response_dir)

    # Save generation parameters
    end = time.time()
    time_elapsed = (end - start) / 60
    params = {
        "context_length": context_length,
        "start_idx": start_idx,
        "max_samples": max_samples,
        "model_name": model_name,
        "batch_size": batch_size,
        "gpu_count": torch.cuda.device_count(),
        "processing_time": time_elapsed,
    }
    params_path = os.path.join(response_dir, 'generation_params.json')
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"Saved generation parameters to {params_path}")
