import os
import yaml
from enum import Enum
from pathlib import Path

CONFIG_FILE: str = "config.yml"
with open(CONFIG_FILE, 'r') as file:
    config = yaml.safe_load(file)

DATA_DIR: str = "data"
PROMPTS_DIR = os.path.join(DATA_DIR, "prompts")
METRICS_DIR = os.path.join(DATA_DIR, "metrics", config['general']['name'])
METRICS_PER_INFERENCE_DIR  = os.path.join(METRICS_DIR, "per_inference")
METRICS_PER_CHUNK_DIR  = os.path.join(METRICS_DIR, "per_chunk")
MODELS_DIR = os.path.join(DATA_DIR, "models", config['general']['name'])
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
SCRIPTS_DIR: str = "scripts"
DIR_LIST = [DATA_DIR, PROMPTS_DIR, METRICS_DIR, MODELS_DIR, DATASET_DIR, METRICS_PER_INFERENCE_DIR, METRICS_PER_CHUNK_DIR]
TOKENIZER_DIR = 'llama2_tokenizer'

_ = list(map(lambda x: os.makedirs(x, exist_ok=True), DIR_LIST))

ENDPOINT_LIST_FPATH:str = os.path.join(MODELS_DIR, "endpoints.json")
REQUEST_PAYLOAD_FPATH:str = os.path.join(PROMPTS_DIR, "payload.jsonl")
RESULTS_FPATH:str = os.path.join(METRICS_DIR, "results.csv")
class TRUNCATE_POLICY(str, Enum):
    AT_PROMPT_TOKEN_LENGTH = 'at-prompt-token-length'

# misc. metrics related
PLACE_HOLDER: int = -1705338041

# metric filenames
COUNTS_FNAME: str = "experiment_counts.csv"
ERROR_RATES_FNAME: str = "error_rates.csv"
RESULTS_DESC_MD_FNAME: str = "results.md"

# plot filenames
ERROR_RATES_PLOT_TEXT: str = "Error rates for different concurrency levels and instance types"
ERROR_RATES_PLOT_FNAME: str = "error_rates.png"
TOKENS_VS_LATENCY_PLOT_TEXT: str = "Tokens vs latency for different concurrency levels and instance types"
TOKENS_VS_LATENCY_PLOT_FNAME: str = "tokens_vs_latency.png"


LATENCY_BUDGET: int = 20

OVERALL_RESULTS_MD: str = """
# Results for performance benchmarking

**Last modified (UTC): {dttm}**

## Summary

The following table provides the best combinations for running inference for different sizes prompts on different instance types.
|Dataset   | Instance type   | Recommendation   |
|---|---|---|
"""

## Dataset=`{dataset}`, instance_type=`{instance_type}`
RESULT_DESC: str = """The best option for staying within a latency budget of `{latency_budget} seconds` on a `{instance_type}` for the `{dataset}` dataset is a `concurrency level of {concurrency}`. A concurrency level of {concurrency} achieves an `average latency of {latency_mean} seconds`, for an `average prompt size of {prompt_size} tokens` and `completion size of {completion_size} tokens` with `{tpm} transactions/minute`."""

RESULT_ROW: str = "|`{dataset}`|`{instance_type}`|{desc}|"

RESULT_FAILURE_DESC: str = """This experiment did not find any combination of concurrency level and other configuration settings that could provide a response within a latency budget of `{latency_budget} seconds` on a `{instance_type}` for the `{dataset}` dataset."""
