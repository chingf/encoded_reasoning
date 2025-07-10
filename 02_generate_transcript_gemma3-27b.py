import torch
from transformers import pipeline
from datasets import load_dataset, Dataset
import time
import os
import json
from config import storage_dir
import argparse

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Generate transcripts using Gemma-3-27B IT model.")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-27b-it", help="Name of the model.")
    parser.add_argument("--context_length", type=int, default=512, help="Maximum context length for generation.")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index for dataset samples.")
    parser.add_argument("--max_samples", type=int, default=5000, help="Maximum number of samples to process.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation.")
    return parser.parse_args()

# Main script
if __name__ == "__main__":
    args = parse_args()

    # Args
    model_name = args.model_name
    context_length = args.context_length
    start_idx = args.start_idx
    max_samples = args.max_samples
    batch_size = args.batch_size

    # Load the dataset
    start = time.time()
    model_storage_dir = os.path.join(storage_dir, "lm_sys", model_name.split("/")[-1])
    if not os.path.exists(model_storage_dir):
        os.makedirs(model_storage_dir)
    cache_dir = "/n/holylfs06/LABS/krajan_lab/Lab/cfang/hf_cache/"
    response_dir = os.path.join(model_storage_dir, f'lm_sys_{start_idx}_{start_idx+max_samples}')
    dataset = Dataset.load_from_disk(os.path.join(storage_dir, 'lm_sys', 'lm_sys_prompts_maxlen=200'))

    # Set up the pipeline with the model and enable multi-GPU usage
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    generator = pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "cache_dir":"/n/holylfs06/LABS/krajan_lab/Lab/cfang/hf_cache/"},
        device_map="auto",  # Automatically distribute across GPUs
        tokenizer=tokenizer,
    )

    prompts = dataset['conversations'][start_idx:start_idx + max_samples]
    outputs = generator(
        prompts,
        do_sample=False,
        max_new_tokens=context_length,
        batch_size=batch_size)

    # Save dataset
    flattened_data = []
    for item in outputs:
        conversation = item[0]['generated_text']
        flattened_data.append({'conversation': conversation})
    new_dataset = Dataset.from_list(flattened_data)
    print(f"Created dataset with {len(new_dataset)} conversations")
    new_dataset.save_to_disk(response_dir)


    # Save processing parameters
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
    print(f"Saved generation parameters to {params_path} with elapsed time: {time_elapsed:.2f} minutes")
