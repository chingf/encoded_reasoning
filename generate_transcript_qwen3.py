import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from tqdm import tqdm
import json
import gc
import os
import torch.multiprocessing as mp
import time

from config import storage_dir

def format_conversation_for_qwen(user_prompt, tokenizer):
    """Format conversation for Qwen3-30B-A3B input with enable_thinking=False"""
    formatted = tokenizer.apply_chat_template(
        user_prompt,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    return formatted

def process_dataset_batch(dataset, tokenizer, model, batch_size=1, max_samples=None, context_length=4096):
    """Process dataset in batches to manage memory and parallelize generation"""
    processed_conversations = []
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing conversations"):
        start = time.time()
        batch = dataset[i:i+batch_size]
        user_prompts = []
        user_turns = []
        for conversation in batch['conversation']:
            user_prompt = conversation[0]
            assert user_prompt['role'] == 'user'
            formatted_prompt = format_conversation_for_qwen([user_prompt], tokenizer)
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
            new_conversation = [user_turns[idx], {'role': 'assistant', 'content': response}]
            processed_conversations.append({'conversation': new_conversation})

        torch.cuda.empty_cache()
        gc.collect()
        end = time.time()
        print(f"Processing this batch took {(end-start)/60} minutes.")
    return processed_conversations

def process_on_gpu(rank, dataset_chunk, batch_size, context_length, model_name, cache_dir, return_dict):
    torch.cuda.set_device(rank)
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map={"": rank},
        cache_dir=cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    end = time.time()
    print(f"Loading model took {(end-start)/60} minutes.")
    results = process_dataset_batch(
        dataset_chunk, tokenizer, model, batch_size=batch_size, context_length=context_length
    )
    return_dict[rank] = results
    del model
    torch.cuda.empty_cache()
    gc.collect()

def process_dataset_batch_multigpu(
        dataset, batch_size=1, context_length=4096,
        start_idx=0, max_samples=None, model_name=None, cache_dir=None):
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs for parallel generation.")
    if max_samples:
        end_index = min(start_idx + max_samples, len(dataset))
        dataset = dataset.select(range(start_idx, end_index))
    chunk_size = len(dataset) // num_gpus
    processes = []
    manager = mp.Manager()
    return_dict = manager.dict()
    for rank in range(num_gpus):
        start = rank * chunk_size
        end = (rank + 1) * chunk_size if rank < num_gpus - 1 else len(dataset)
        dataset_chunk = dataset.select(range(start, end))
        p = mp.Process(
            target=process_on_gpu,
            args=(rank, dataset_chunk, batch_size, context_length, model_name, cache_dir, return_dict)
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    all_results = []
    for rank in range(num_gpus):
        all_results.extend(return_dict[rank])
    return all_results

def main():
    mp.set_start_method('spawn', force=True)

    # Arguments
    #model_name = "Qwen/Qwen3-30B-A3B"
    #context_length = 512
    #start_idx = 0
    #max_samples = 10500
    #batch_size = 128

    model_name = "Qwen/Qwen3-14B"
    context_length = 512
    start_idx = 0
    max_samples = 10500
    batch_size = 128

    # Directory handling and loading data
    start = time.time()
    model_storage_dir = os.path.join(storage_dir, "lm_sys", model_name.split("/")[-1])
    if not os.path.exists(model_storage_dir):
        os.makedirs(model_storage_dir)
    cache_dir = "/n/holylfs06/LABS/krajan_lab/Lab/cfang/hf_cache/"
    ds = load_dataset("lmsys/lmsys-chat-1m")

    # Actually run processing
    processed_data = process_dataset_batch_multigpu(
        ds['train'],
        batch_size=batch_size,
        context_length=context_length,
        start_idx=start_idx,
        max_samples=max_samples,
        model_name=model_name,
        cache_dir=cache_dir,
    )
    print(f"Processed {len(processed_data)} conversations.")

    # Save new dataset
    new_dataset = Dataset.from_list(processed_data)
    print(f"Created dataset with {len(new_dataset)} conversations")
    new_dataset.save_to_disk(os.path.join(model_storage_dir, 'lm_sys_responses'))

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
    params_path = os.path.join(model_storage_dir, 'lm_sys_responses', 'generation_params.json')
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"Saved generation parameters to {params_path}")

if __name__ == "__main__":
    main()