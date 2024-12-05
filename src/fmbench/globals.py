import os
import yaml
import boto3
import requests
import tempfile
from enum import Enum
from pathlib import Path
from fmbench import defaults
from datetime import datetime
from typing import Optional, Dict
import importlib.resources as pkg_resources


FMBENCH_PACKAGE_NAME: str = "fmbench"

# This is the hf prefix to the source data file which acts as an identifier to whether
# the dataset is a hugging face dataset or not
HF_DATASET_PREFIX: str = "hf:"
DEFAULT_IMAGE_FORMAT: str = "JPEG"

# This is the ds_N default value of the number of rows to be processed from the hf dataset.
# If this value is already given in the 'dataset' section of the config file, that will be used.
DEFAULT_HF_DS_N_VALUE: int = 100

current_working_directory: str = Path.cwd()

CONFIG_FILEPATH_FILE: str = current_working_directory / 'config_filepath.txt'

PRICING_FALLBACK_YAML_PATH="https://raw.githubusercontent.com/aws-samples/foundation-model-benchmarking-tool/refs/heads/main/src/fmbench/configs/pricing_fallback.yml"

# S3 client initialization
s3_client = boto3.client('s3')
session = boto3.session.Session()
region_name = session.region_name
if region_name is None:
    print(f"boto3.session.Session().region_name is {region_name}, "
          f"going to use an metadata api to determine region name")
    # THIS CODE ASSUMED WE ARE RUNNING ON EC2, for everything else
    # the boto3 session should be sufficient to retrieve region name
    resp = requests.put("http://169.254.169.254/latest/api/token",
                        headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"})
    token = resp.text
    region_name = requests.get("http://169.254.169.254/latest/meta-data/placement/region",
                               headers={"X-aws-ec2-metadata-token": token}).text
    print(f"region_name={region_name}, also setting the AWS_DEFAULT_REGION env var")
    os.environ["AWS_DEFAULT_REGION"] = region_name
print(f"region_name={region_name}")

# Configuring the role ARN -- extract the role name
caller = boto3.client('sts').get_caller_identity()
account_id = caller.get('Account')
role_arn_from_env = os.environ.get('FMBENCH_ROLE_ARN')
if role_arn_from_env:
    print(f"role_arn_from_env={role_arn_from_env}, using it to set arn_string")
    arn_string = role_arn_from_env
else:
    print(f"role_arn_from_env={role_arn_from_env}, using current sts caller identity to set arn_string")
    arn_string = caller.get('Arn')
    # if this is an assumed role then remove the assumed role related pieces
    # because we are also using this role for deploying the SageMaker endpoint
    # arn:aws:sts::015469603702:assumed-role/SSMDefaultRoleForOneClickPvreReporting/i-0c5bba16a8b3dac51
    # should be converted to arn:aws:iam::015469603702:role/SSMDefaultRoleForOneClickPvreReporting
    if ":assumed-role/" in arn_string:
        role_name = arn_string.split("/")[-2]
        arn_string = f"arn:aws:iam::{account_id}:role/{role_name}"
        print(f"the sts role is an assumed role, setting arn_string to {arn_string}")
    else:
        arn_string = caller.get('Arn')

ROLE_NAME = arn_string.split('/')[-1]

current_config_file = os.environ.get("CONFIG_FILE_FMBENCH")
# if var is true, use that from cli
if current_config_file is not None:
    CONFIG_FILE = current_config_file
    print(f"config file current -> {CONFIG_FILE}, {current_config_file}")
else:
    CONFIG_FILE = Path(CONFIG_FILEPATH_FILE).read_text().strip()
    print(f"config file current -> {CONFIG_FILE}, {current_config_file}")


if CONFIG_FILE.startswith("s3://"):
    # Parse the S3 URL
    bucket_name, key_path = CONFIG_FILE.replace("s3://", "").split("/", 1)
    # Use boto3 to access the S3 service
    s3 = boto3.resource('s3')
    # Fetch the object from S3
    obj = s3.Object(bucket_name, key_path)
    # Read the object's content
    CONFIG_FILE_CONTENT = obj.get()['Body'].read().decode('utf-8')
elif CONFIG_FILE.startswith("https://"):
    # Fetch the content from the HTTPS URL
    response = requests.get(CONFIG_FILE)
    response.raise_for_status()  # Ensure we got a successful response
    CONFIG_FILE_CONTENT = response.text
else:
    CONFIG_FILE_CONTENT = Path(CONFIG_FILE).read_text()

# check if the file is still parameterized and if so replace the parameters with actual values
# if the file is not parameterized then the following statements change nothing
write_bucket = os.environ.get("WRITE_BUCKET", f"{defaults.DEFAULT_BUCKET_WRITE}-{region_name}-{account_id}")
# check if the tmp dir is used as an argument if local mode is set to yes. If so, then use that as the temp file directory
# else use the default `tempfile` option
tmp_dir = os.environ.get("TMP_DIR", tempfile.gettempdir())
args = dict(region=session.region_name,
            role_arn=arn_string,
            read_tmpdir=os.path.join(tmp_dir, defaults.DEFAULT_LOCAL_READ),
            write_tmpdir=os.path.join(tmp_dir, defaults.DEFAULT_LOCAL_WRITE),
            write_bucket=write_bucket,
            read_bucket=f"{defaults.DEFAULT_BUCKET_READ}-{region_name}-{account_id}")
CONFIG_FILE_CONTENT = CONFIG_FILE_CONTENT.format(**args)

# Load the configuration
config = yaml.safe_load(CONFIG_FILE_CONTENT)
local_mode = os.environ.get("LOCAL_MODE")
if local_mode == "yes":
    print("globals.py, local_mode = yes")
    config['aws']['s3_and_or_local_file_system'] = 'local'
    config['s3_read_data']['s3_or_local_file_system'] = 'local'
    if config['s3_read_data'].get('local_file_system_path') is None:
        config['s3_read_data']['local_file_system_path'] = os.path.join(tmp_dir, defaults.DEFAULT_LOCAL_READ)
    if config['aws'].get('local_file_system_path') is None:
        config['aws']['local_file_system_path'] = os.path.join(tmp_dir, defaults.DEFAULT_LOCAL_WRITE)

# iterate through each experiment and populate the parameters section in the inference spec
for i in range(len(config['experiments'])):
    # for the experiment at index i, look up the parameter set
    # retrieve the parameter set from the inference_parameter section
    # assign the parameters from that parameter set to a new key called
    # parameters in that experiment
    parameters = config['inference_parameters'][config['experiments'][i]['inference_spec']['parameter_set']]
    config['experiments'][i]['inference_spec']['parameters'] = parameters
    if config['experiments'][i].get('bucket') is None:
        config['experiments'][i]['bucket'] = config['aws']['bucket']
print(f"loaded config: {config}")


# get the model evaluation configuration file which contains information on the 
# ground truth, the method name, and directory structure being used
eval_config: Optional[Dict] = None
config_dir = Path(pkg_resources.files('fmbench'), 'configs')
# load the model evaluation configuration file based on the ground truth, 
# formatted into it from the main config file if any
if 'model_evaluations' in config and config['model_evaluations'] is not None:
    model_evaluation_common_file: str = config['model_evaluations']
    ground_truth_col_key: Optional[str] = config['datasets'].get('ground_truth_col_key', None)
    eval_module = Path(model_evaluation_common_file)
    eval_file_path: str = os.path.join(config_dir, eval_module)
    if ground_truth_col_key is not None:
        with open(eval_file_path, 'r') as file:
            model_eval_info = file.read()
            # load the preliminary unformatted config file to fetch the method name and plug it into
            # the prompt template file names
            eval_config = yaml.safe_load(model_eval_info)
            print(f"loaded eval configuration file: {eval_config}")
    else:
        eval_config=None
        print(f"Evalaution configuration file not found in the config file. Provide a valid eval configuration file name.")

# data directory and prompts
PER_ACCOUNT_DIR: str = f"{config['general']['name']}-{ROLE_NAME}"
DATA_DIR: str = os.path.join(PER_ACCOUNT_DIR, config['dir_paths']['data_prefix'])
PROMPTS_DIR = os.path.join(DATA_DIR, config['dir_paths']['prompts_prefix'])

# Metrics directory based on date and time
current_time = datetime.now()

# Assuming current_time is a datetime object
formatted_time = current_time.strftime("%Y/%m/%d/%H/%M")

# Split the formatted_time into components
year, month, day, hour, minute = formatted_time.split('/')

# Construct the METRICS_DIR path
METRICS_DIR = f"{DATA_DIR}/metrics/yyyy={year}/mm={month}/dd={day}/hh={hour}/mm={minute}"

METRICS_PER_INFERENCE_DIR = os.path.join(METRICS_DIR, "per_inference")
METRICS_PER_CHUNK_DIR = os.path.join(METRICS_DIR, "per_chunk")
METRICS_PER_POLL_EVAL_DIR_NAME: str = "per_poll_eval"

METRICS_PER_INFERENCE_DIR = os.path.join(METRICS_DIR, "per_inference")
METRICS_PER_CHUNK_DIR = os.path.join(METRICS_DIR, "per_chunk")
ENDPOINT_METRICS_FNAME = "endpoint_metrics.csv"
ENDPOINT_METRICS_SUMMARIZED_FNAME = "endpoint_metrics_summarized.csv"

# These are the column names that are present in the SageMaker and EC2 instance utilization metrics
# SageMaker utilization metrics contain an EndpointName column
SAGEMAKER_EP_NAME_COL: str = "EndpointName"
# These are the list of quantiles that are measured as a part of the metric utilization calculation
# for EC2 benchmarking
UTILIZATION_QUANTILE_METRICS = [0, 0.25, 0.5, 0.75, 1.0]

# Models directory based on date and time 
MODELS_DIR = f"{DATA_DIR}/models"

# Use this to upload to the s3 bucket (extracted from the config file)
BUCKET_NAME = config['aws']['bucket']
READ_BUCKET_NAME = config['s3_read_data']['read_bucket']

# S3 prefix
PREFIX_NAME = config['dir_paths']['data_prefix']

# SOURCE data is where your actual data resides in s3
SOURCE_DATA = config['s3_read_data']['source_data_prefix']

# Read the prompt template that the user uploads
PROMPT_TEMPLATE_S3_PREFIX = config['s3_read_data']['prompt_template_dir']

# Initialize the scripts directory
SCRIPTS_DIR: str = "fmbench/scripts"

# Contruct the path to the evaluation prompt and the different rules in 
# the rules directory for respective subjective eval criteria
if eval_config is not None:
    EVAL_PROMPT_TEMPLATES: str = os.path.join(PROMPT_TEMPLATE_S3_PREFIX,
                                              eval_config['model_evaluations']['model_eval_dir'].get('eval_prompts_dir', None))
    EVAL_DIR: str = eval_config['model_evaluations']['model_eval_dir'].get('eval_prompts_dir', None)
    EVAL_INSTRUCTIONS_DIR: str = eval_config['model_evaluations']['model_eval_dir'].get('eval_instructions_dir', None)

# METADATA DIR TO HANDLE DYNAMIC S3 PATHS FOR METRICS/RESULTS
METADATA_DIR:str = config['dir_paths']['metadata_dir']
METRICS_PATH_FNAME: str = "metrics_path.txt"

# Name of the .txt file where the HF token is stored
HF_TOKEN_FNAME: str = "hf_token.txt"

DIR_LIST = [DATA_DIR, PROMPTS_DIR, METRICS_DIR, MODELS_DIR, METRICS_PER_INFERENCE_DIR, METRICS_PER_CHUNK_DIR]

# this is for custom tokenizers
TOKENIZER_DIR_S3 = config['s3_read_data']['tokenizer_prefix']
TOKENIZER = 'tokenizer'
# we take the first experiment's model id as the model whose tokenizer is used for 
# counting tokens on the dataset. This is done just for ease of coding, this is just
# token counting logic on the client side (does not impact the tokenizer the model uses)
# NOTE: if tokenizer files are provided in the tokenizer directory then they take precedence
# if the files are not present then we load the tokenizer for this model id from Hugging Face
TOKENIZER_MODEL_ID = config['experiments'][0].get('model_id')

# If the model that is being benchmarked does not necessarilly have an HF model id, but you
# want to use the hf tokenizer for that model, then mention the "hf_tokenizer_model_id" in the
# experiments section to load the tokenizer at runtime. The alternative option to this is
# to provide the "config.json" and "tokenizer.json" files in the read directory. If neither are provided, 
# then the default 750-1000 tokens tokenizer will be used.
HF_TOKENIZER_MODEL_ID = config['experiments'][0].get('hf_tokenizer_model_id')

DEPLOYMENT_SCRIPT_S3 = config['s3_read_data']['scripts_prefix']

_ = list(map(lambda x: os.makedirs(x, exist_ok=True), DIR_LIST))

# Define the endpoint list as the config-general name plus the role arn for unique generation 
# from different roles in the same/different accounts
ENDPOINT_LIST_PATH: str = os.path.join(MODELS_DIR, "endpoints.json")

REQUEST_PAYLOAD_FPATH: str = os.path.join(PROMPTS_DIR, "payload.jsonl")
RESULTS_FPATH: str = os.path.join(METRICS_DIR, "results.csv")


class TRUNCATE_POLICY(str, Enum):
    AT_PROMPT_TOKEN_LENGTH = 'at-prompt-token-length'

# misc. metrics related
PLACE_HOLDER: int = -1705338041
RESULTS_DIR: str = f"results-{config['general']['name']}"

# benchmarking - metric filenames
COUNTS_FNAME: str = "experiment_counts.csv"
ERROR_RATES_FNAME: str = "error_rates.csv"
RESULTS_DESC_MD_FNAME: str = "report.md"
SUMMARY_METRICS_W_PRICING_FNAME: str = "summary_metrics_w_pricing.csv"
INSTANCE_PRICING_PER_HOUR_FNAME: str = "instance_pricing_per_hour.csv"
SUMMARY_METRICS_FOR_DATASET_W_SCORES_FNAME: str = "summary_metrics_for_dataset_w_scores.csv"
SUMMARY_METRICS_FOR_DATASET_W_SCORES_BEST_OPTION_FNAME: str = "summary_metrics_for_dataset_best_option.csv"
SUMMARY_METRICS_FOR_DATASET_W_SCORES_BEST_OPTION_EACH_INSTANCE_TYPE_FNAME: str = "summary_metrics_for_dataset_best_option_each_instance_type.csv"
SUMMARY_MODEL_ENDPOINT_COST_PER_INSTANCE: str = "endpoint_per_instance_per_run_costs.csv"
BUSINESS_SUMMARY_PLOT_FNAME: str = "business_summary.png"
BUSINESS_SUMMARY_PLOT_FNAME2: str = "business_summary_barchart.png"
LATENCY_CHART_PLOT_FNAME: str = "latency_summary_chart.png"

# evaluation - metric filenames
PER_INFERENCE_FILE_WITH_COSINE_SIMILARITY_SCORES: str = "per_inference_quantitative_eval_metrics.csv"
EVAL_COL_SUFFIX: str = '_eval_prompt'
PROCESSED_EVAL_PROMPT_PAYLOADS: str = "processed_eval_prompts_for_inference.csv"
MODEL_EVALUATION_JUDGE_COMPLETIONS_DIR: str = "judge_model_eval_completions"
MODEL_EVAL_COMPLETIONS_CSV: str = "raw_llm_as_a_judge_evals.csv"
LLM_JUDGE_PANEL_RESPONSE_SUMMARIES: str = "llm_as_a_judge_per_eval_summary.csv"
# this csv contains all of the incorrect verdict responses from the PoLL
# evaluation of responses using Max Voting. View this csv to get more insight
# into where the model went wrong, and what to fix
VERDICT_TYPE_BREAKDOWN_FOR_CORRECT_FILE: str = "verdict_type_breakdown_for_correct.csv"
VERDICT_TYPE_BREAKDOWN_FOR_INCORRECT_FILE: str = "verdict_type_breakdown_for_incorrect.csv"
PER_MODEL_ACCURACY_W_VERDICT_TYPE_FILE: str = "per_model_accuracy_w_verdict_type.csv"
MAJORITY_VOTE_DF_RAW_RESULTS_FILE: str = "majority_vote_results_raw.csv"
PER_PAYLOAD_MODEL_ACCURACY_MAJORITY_VOTING: str = "per_payload_model_accuracy_majority_vote.csv"
PER_MODEL_ACCURACY_PER_EVAL_JUDGE: str = "per_model_per_eval_judge_accuracy.csv"
CANDIDATE_MODEL_ACCURACY_FILE: str = "candidate_model_accuracy.csv"
INCORRECT_VERDICT_RESPONSES_FILE: str = "incorrect_verdict_responses.csv"
CORRECT_VERDICT_RESPONSES_FILE: str = "correct_verdict_responses.csv"
SCORING_RESULT_COUNT_POLL: str = "PoLL_result_count_correct_incorrect.csv"
PER_MODEL_ACCURACY_POLL: str = "PoLL_per_model_accuracy.csv"
PER_PAYLOAD_PER_MODEL_POLL_ACCURACY: str = "majority_vote_accuracy_per_payload_file.csv"
EVAL_COST_PER_JUDGE_MODEL: str = "eval_cost_per_llm_evaluator.csv"
# contains all tt data of the LLM completion from the evaluation process
ALL_EVALUATIONS_IN_TXT: str = "all_judges_evals.txt"
# contains the final analysis done by a final LLM in the loop to summarize
# all evaluations done by panel of LLM evaluators on candidate model responses
NEEDS_FURTHER_EVAL_FILE: str = "responses_need_further_eval.txt"
# accuracy charts 
PER_PAYLOAD_FILE_ACCURACY_TRAJECTORY: str = "accuracy_trajectory_per_payload.png"
OVERALL_CANDIDATE_MODEL_MAJORITY_VOTING_ACCURACY: str = "overall_candidate_model_majority_voting_accuracy.png"

# plot filenames
ERROR_RATES_PLOT_TEXT: str = "Error rates for different concurrency levels and instance types"
ERROR_RATES_PLOT_FNAME: str = "error_rates.png"
TOKENS_VS_LATENCY_PLOT_TEXT: str = "Tokens vs latency for different concurrency levels and instance types"
TOKENS_VS_LATENCY_PLOT_FNAME: str = "tokens_vs_latency.png"
CONCURRENCY_VS_INFERENCE_LATENCY_PLOT_FNAME: str = "concurrency_vs_inference_latency.png"
CONCURRENCY_VS_INFERENCE_LATENCY_PLOT_TEXT: str = "Concurrency Vs latency for different instance type for selected dataset"
LATENCY_BUDGET: int = 5

OVERALL_RESULTS_MD: str = """
# {title}

|**Last modified (UTC)** | **FMBench version**  |
|---|---|
|{dttm}|{fmbench_version}|


## Summary

{business_summary}

## Per instance results

The following table provides the best combinations for running inference for different sizes prompts on different instance types. The following dataset(s) were used for this test: {datasets}.

|Dataset   | Instance type   | Recommendation   |
|---|---|---|
"""

# Dataset=`{dataset}`, instance_type=`{instance_type}`
RESULT_DESC: str = """The best option for staying within a latency budget of `{latency_budget} seconds` on a `{instance_type}` for the `{dataset}` dataset is a `concurrency level of {concurrency}`. A concurrency level of {concurrency} achieves an `median latency of {latency_median} seconds`, for an `average prompt size of {prompt_size} tokens` and `completion size of {completion_size} tokens` with `{tpm} transactions/minute`."""

RESULT_ROW: str = "|`{dataset}`|`{instance_type}`|{desc}|"

RESULT_FAILURE_DESC: str = """This experiment did not find any combination of concurrency level and other configuration settings that could provide a response within a latency budget of `{latency_budget} seconds` on a `{instance_type}` for the `{dataset}` dataset."""

PROMPT_TEMPLATE: str = """<s>[INST] <<SYS>>
You are an assistant for question-answering tasks. Use the following pieces of retrieved context in the section demarcated by "```" to answer the question. If you don't know the answer just say that you don't know. Use three sentences maximum and keep the answer concise.
<</SYS>>

```
{context}
```

Question: {input}

[/INST]
Answer:

"""
