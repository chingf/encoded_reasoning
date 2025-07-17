import torch
import numpy as np
from transformers import pipeline
from datasets import load_dataset, Dataset
import time
import os
import json
from config import storage_dir
from transformers import AutoTokenizer
from config import storage_dir, hf_cache_dir

# Args
model_name = "meta-llama/Llama-3.3-70B-Instruct"  # For tokenizer
max_prompt_token_length = 200

# Load the dataset and tokenizer
dataset = load_dataset("lmsys/lmsys-chat-1m")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Count tokens and identify promps within the limit
def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=False))
print('Counting tokens in conversation prompts...')
counts = [count_tokens(c[:1][0]['content'], tokenizer) for c in dataset['train']['conversation']]
counts = np.array(counts)
idxs = np.argwhere(counts <=max_prompt_token_length)

# Take subset of data and save it
print('Taking a subset of the dataset...')
dataset_subset = dataset['train'][idxs]
print('Extracting only the prompts...')
dataset_subset = [c[:1] for c in dataset_subset['conversation']]
print('Converting to Dataset format and saving...')
dataset_subset = Dataset.from_dict({"conversations": dataset_subset})
save_path = os.path.join(storage_dir, f'lm_sys_prompts_maxlen={max_prompt_token_length}')
dataset_subset.save_to_disk(save_path)