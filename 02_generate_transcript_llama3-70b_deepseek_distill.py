import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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
    parser.add_argument("--context_length", type=int, default=1200, help="Maximum context length for generation.")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index for dataset samples.")
    parser.add_argument("--max_samples", type=int, default=3000, help="Maximum number of samples to process.")
    parser.add_argument("--batch_size", type=int, default=250, help="Batch size for generation.")
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
    model_storage_dir = os.path.join(storage_dir, "lm_sys", model_name.split("/")[-1])
    if not os.path.exists(model_storage_dir):
        os.makedirs(model_storage_dir)
    response_dir = os.path.join(model_storage_dir, f'lm_sys_{start_idx}_{start_idx+max_samples}')
    dataset = Dataset.load_from_disk(os.path.join(storage_dir, 'lm_sys', 'lm_sys_prompts_maxlen=200'))

    # Configure 4-bit quantization with optimal settings
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Use NF4 quantization (better than fp4)
        bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bfloat16 for better performance
        bnb_4bit_use_double_quant=True,  # Double quantization for further memory savings
    )

    # Set up tokenizer with memory-efficient settings
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        padding_side='left',
        cache_dir=hf_cache_dir,
        use_fast=True,  # Use fast tokenizer for better performance
    )
    
    # Add pad token if missing (common issue with some models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with optimized settings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,  # Use the BitsAndBytesConfig
        device_map="auto",  # Automatically distribute across GPUs
        cache_dir=hf_cache_dir,
        torch_dtype=torch.bfloat16,  # Keep for non-quantized parts
        low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
        trust_remote_code=True,  # May be needed for some models
    )

    # Create pipeline with memory-efficient settings
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        batch_size=batch_size,
        model_kwargs={
            "use_cache": True,  # Enable KV caching for efficiency
            "pad_token_id": tokenizer.pad_token_id,
        }
    )

    start = time.time()
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

    # Load to huggingface (for debugging)
    #hf_url = f"chingfang17/test_llama_deepseek_distill" 
    #new_dataset.push_to_hub(hf_url)

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
