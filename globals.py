import os
from enum import Enum
from pathlib import Path

CONFIG_FILE: str = "config.yml"
DATA_DIR: str = "data"
PROMPTS_DIR = os.path.join(DATA_DIR, "prompts")
METRICS_DIR = os.path.join(DATA_DIR, "metrics")
MODELS_DIR = os.path.join(DATA_DIR, "models")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
DIR_LIST = [DATA_DIR, PROMPTS_DIR, METRICS_DIR, MODELS_DIR, DATASET_DIR]
TOKENIZER_DIR = 'llama2_tokenizer'

_ = list(map(lambda x: os.makedirs(x, exist_ok=True), DIR_LIST))

ENDPOINT_LIST_FPATH:str = os.path.join(MODELS_DIR, "endpoints.json")
REQUEST_PAYLOAD_FPATH:str = os.path.join(PROMPTS_DIR, "payload.jsonl")
RESULTS_FPATH:str = os.path.join(METRICS_DIR, "results.csv")
class TRUNCATE_POLICY(str, Enum):
    AT_PROMPT_TOKEN_LENGTH = 'at-prompt-token-length'    
