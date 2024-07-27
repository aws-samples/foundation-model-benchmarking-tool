from huggingface_hub import snapshot_download
from pathlib import Path
import os

# - This will download the model into the current directory where ever the jupyter notebook is running
local_model_path = Path("hf/")
local_model_path.mkdir(exist_ok=True)
model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
# Only download pytorch checkpoint files
allow_patterns = ["*.json", "*.txt", "*.model", "*.safetensors", "*.bin", "*.chk", "*.pth"]

# - Leverage the snapshot library to donload the model since the model is stored in repository using LFS
model_download_path = snapshot_download(
    repo_id=model_name, 
    cache_dir=local_model_path, 
    allow_patterns=allow_patterns, 
    ## Replace token value with your own token from your HuggingFace Account
    ## following token is invalid and will not work
    token='YOUR_HF_TOKEN_GOES_HERE'
)