import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import gc
import os
from datasets import load_from_disk
import numpy as np
import matplotlib.pyplot as plt
import re
from config import storage_dir
import re
import codecs


# Arguments
model_name = "Qwen/Qwen3-14B"
thinking = True
shorten_to = 4096

#model_name = "meta-llama/Llama-3.3-70B-Instruct"
#thinking = False
#shorten_to = 800

# model_name = "google/gemma-3-27b-it"
# thinking = False
# shorten_to = 1024


# Set Paths
dset_name = 'lm_sys_responses'
if thinking:
    dset_name += '_thinking'
base_model_name = model_name.split("/")[-1]
model_storage_dir = os.path.join(storage_dir, "lm_sys", model_name.split("/")[-1])
response_path = os.path.join(model_storage_dir, dset_name)


# Loading dataset and exclude bad prompts
dataset = load_from_disk(response_path)
print(dataset)
filtered_dataset = [
    {'conversation': row} for row in dataset['conversation']
    if 'single dot' not in row[0]['content']
]
filtered_hf_dataset = Dataset.from_list(filtered_dataset)

generation_params_path = os.path.join(response_path, 'generation_params.json')
if os.path.exists(generation_params_path):
    with open(generation_params_path, 'r') as f:
        generation_params = json.load(f)
    print(generation_params)
    
def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=False))

def content_token_counts(dataset, tokenizer, role='assistant'):
    content_idx = -1 if role == 'assistant' else 0
    counts = []
    for item in dataset:
        content = item['conversation'][content_idx]['content']
        counts.append(count_tokens(content, tokenizer))
    return np.array(counts)

tokenizer = AutoTokenizer.from_pretrained(model_name)


# Conversion to ROT13
def rot13_alpha(text):
    # Split text around '<think>' and '</think>'
    segments = re.split(r'(<think>|</think>)', text)
    converted_segments = []

    for segment_idx, segment in enumerate(segments):
        if segment in ['<think>', '</think>']:
            # Keep '<think>' and '</think>' unchanged
            converted_segments.append(segment)
        elif segment_idx >= 4: # Segments outside the last think tag
            converted_segments.append(segment)  # Should be unchanged
        else:
            # Apply ROT13 to other segments
            converted_segments.append(codecs.encode(segment, 'rot_13'))
    

    # Reassemble the text
    reassembled_text = ''.join(converted_segments)
    return reassembled_text

def convert_dataset_to_rot13(dataset):
    new_dataset = []
    n_items = 0
    for item in dataset:
        new_item = item.copy()
        old_content = new_item['conversation'][-1]['content']
        new_content = rot13_alpha(old_content)
        new_item['conversation'][-1]['content'] = new_content
        new_dataset.append(new_item)
        n_items += 1
    return new_dataset

rot13_dataset = convert_dataset_to_rot13(filtered_hf_dataset)


# Truncate assistant content
def truncate_assistant_content(new_item, tokenizer, shorten_to):
    user_content = new_item['conversation'][0]['content']
    assistant_content = new_item['conversation'][-1]['content']

    user_tokens = tokenizer.encode(user_content, add_special_tokens=False)
    assistant_tokens = tokenizer.encode(assistant_content, add_special_tokens=False)

    total_tokens = len(user_tokens) + len(assistant_tokens)

    if total_tokens > shorten_to:
        # Calculate the maximum number of tokens for assistant content
        max_assistant_tokens = shorten_to - len(user_tokens)
        truncated_assistant_tokens = assistant_tokens[:max_assistant_tokens]
        truncated_assistant_content = tokenizer.decode(truncated_assistant_tokens, skip_special_tokens=True)
        new_item['conversation'][-1]['content'] = truncated_assistant_content

    return new_item

# Drop rows from new_dataset that are identical to filtered_hf_dataset
filtered_and_truncated_rot13_dataset = []
dropped_data = []
for new_item, original_item in zip(rot13_dataset, filtered_hf_dataset):
    if new_item['conversation'][-1]['content'] == original_item['conversation'][-1]['content']:
        dropped_data.append(new_item)
    else:
        if shorten_to is not None:
            new_item = truncate_assistant_content(new_item, tokenizer, shorten_to)
        filtered_and_truncated_rot13_dataset.append(new_item)
        
new_hf_dataset = Dataset.from_list(filtered_and_truncated_rot13_dataset)
if shorten_to is not None:
    rot13_save_path = os.path.join(
        model_storage_dir, f'{dset_name}_rot13_clip{shorten_to}')
else:
    rot13_save_path = os.path.join(
        model_storage_dir, f'{dset_name}_rot13')
new_hf_dataset.save_to_disk(rot13_save_path)


# Load to huggingface
hf_url = f"chingfang17/{base_model_name}_{dset_name}_rot13" 
if shorten_to is not None:
    hf_url += f"_clip{shorten_to}"
new_hf_dataset.push_to_hub(hf_url)