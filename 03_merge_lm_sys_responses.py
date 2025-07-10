import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import os
import re
from datasets import load_from_disk
from datasets import concatenate_datasets
from config import storage_dir

# Arguments
model_name = "Qwen3-14B"
thinking = True

#model_name = "meta-llama/Llama-3.3-70B-Instruct"
#thinking = False

# model_name = "google/gemma-3-27b-it"
# thinking = False


# Set paths
model_storage_dir = os.path.join(storage_dir, "lm_sys", model_name.split("/")[-1])
response_paths = os.path.join(model_storage_dir, 'lm_sys_responses')

# Processing
# Step 1: Collect directories matching the format "lm_sys_{start_num}_{end_num}"
pattern = r"lm_sys_(\d+)_(\d+)"
if thinking:
    pattern += "_thinking"
directories = []

for dir_name in os.listdir(model_storage_dir):
    match = re.match(pattern, dir_name)
    if match:
        start_num, end_num = map(int, match.groups())
        size = end_num - start_num
        directories.append((dir_name, start_num, end_num, size))
        
# Step 2: Sort directories by start_num
directories.sort(key=lambda x: x[1])

# Step 3: Load datasets, index them, and merge into one dataset
datasets_to_merge = []

for dir_name, start_num, end_num, size in directories:
    dataset_path = os.path.join(model_storage_dir, dir_name)
    dataset = load_from_disk(dataset_path)
    
    # Index the dataset from 0 to size
    dataset = dataset.select(range(size))
    
    # Add the dataset to the list for merging
    datasets_to_merge.append(dataset)

# Merge all datasets into one
merged_dataset = concatenate_datasets(datasets_to_merge)

# Save the merged dataset if needed
dset_name = 'lm_sys_responses'
if thinking:
    dset_name += '_thinking'
merged_dataset.save_to_disk(os.path.join(
    model_storage_dir, dset_name))