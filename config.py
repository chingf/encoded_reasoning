import os

is_kempner_cluster = os.path.exists('/n/holylfs06')
if not is_kempner_cluster:
    storage_dir = '/workspace/data/storage'
    hf_cache_dir = '/workspace/data/huggingface-cache/hub'
    os.environ['HF_HUB_CACHE'] = hf_cache_dir
else:
    storage_dir = '/n/holylfs06/LABS/krajan_lab/Lab/cfang/encoded_reasoning'
    hf_cache_dir = '/n/holylfs06/LABS/krajan_lab/Lab/cfang/hf_cache'