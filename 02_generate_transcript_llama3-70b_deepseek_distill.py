import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
import time
import os
import json
from config import storage_dir, hf_cache_dir
import argparse

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Generate transcripts using DeepSeek Distill Llama-70B-Instruct model.")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-70B", help="Name of the model.")
    parser.add_argument("--context_length", type=int, default=2048, help="Maximum context length for generation.")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index for dataset samples.")
    parser.add_argument("--max_samples", type=int, default=6000, help="Maximum number of samples to process.")
    parser.add_argument("--batch_size", type=int, default=2000, help="Batch size for generation.")
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
    response_dir = os.path.join(model_storage_dir, f'lm_sys_{start_idx}_{start_idx+max_samples}')
    dataset = Dataset.load_from_disk(os.path.join(storage_dir, 'lm_sys', 'lm_sys_prompts_maxlen=200'))

    # Set up the pipeline with the model and enable multi-GPU usage
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

    #generator = pipeline(  # Old way, 16-bits
    #    "text-generation",
    #    model=model_name,
    #    model_kwargs={
    #        "torch_dtype": torch.bfloat16,
    #        "cache_dir":hf_cache_dir,
    #    },
    #    device_map="auto",  # Automatically distribute across GPUs
    #    tokenizer=tokenizer,
    #)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=hf_cache_dir,
        load_in_4bit=True,  # Enable 4-bit quantization
    )
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    prompts = dataset['conversations'][start_idx:start_idx + max_samples]
    print("passing prompts to generator")
    outputs = generator(
        prompts,
        do_sample=False,
        max_new_tokens=context_length,
        batch_size=batch_size)

    # Save dataset
    flattened_data = []
    for item in outputs:
        conversation = item[0]['generated_text']
        conversation[-1]['content'] = '<think>\n' + conversation[-1]['content'] 
        flattened_data.append({'conversation': conversation})
    new_dataset = Dataset.from_list(flattened_data)
    print(f"Created dataset with {len(new_dataset)} conversations")
    new_dataset.save_to_disk(response_dir)

    # Load to huggingface
    hf_url = f"chingfang17/test_llama_deepseek_distill" 
    new_dataset.push_to_hub(hf_url)

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