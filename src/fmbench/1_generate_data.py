#!/usr/bin/env python
# coding: utf-8

# # Generate Data: Gather data, create prompts/payloads of different sizes
# ---------
# *This notebook works best with the conda_python3 kernel on a ml.t3.medium machine*.
# 
# ### This part of our solution design includes 
# 
# - running and downloading our specific dataset
# 
# - generating prompts as payloads of different sizes that we will send to our different model endpoints with different combinations of concurrency levels that we will later use to run inference and generate benchmarking metrics and visualizations.
# 
# #### This file will generate all data on wikiqa (english version) with prompt sizes 300 - 4000 token lengths in different payload sizes to send to the model endpoint during the inference pipeline. You will also be able to generate the normal wikiqa dataset from the actual 'long bench dataset'. This notebook then focuses on 3 main deliverables:
# 
# 1. Loading the dataset that is stored within the dataset in the data directory.
# 
# 
# 2. Generating payloads: This notebook also converts the loaded datasets into payloads based on the input question and records teh context length of the prompt to send as a part of the payload during running inferences on the deployed endpoints.
# 
#     - All of the prompts are saved in this data directory in a file named all_prompts.csv.
#     
# 
# 3. Constructing different sized payloads

# #### Import all of the necessary libraries below to run this notebook
import io
import sys
import copy
import json
import logging
import itertools
import pandas as pd
import importlib.util
from pathlib import Path
from fmbench.utils import *
from fmbench.globals import *
from typing import Dict, List
import importlib.resources as pkg_resources
from botocore.exceptions import ClientError

## setting up a logger to track for all of the runs
logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

## config.yml file contains information that is used across this benchmarking environment, 
## such as information about the aws account, prompts, payloads to be used for invocations
config = load_config(CONFIG_FILE)
logger.info(json.dumps(config, indent=2))


# #### Define the file path for the prompt template
s3_file_path = "/".join([config['s3_read_data']['prompt_template_dir'],
                         config['s3_read_data']['prompt_template_file']])

## download the file from s3 else check locally and use that version
prompt_template: str = read_from_s3(config['s3_read_data']['read_bucket'], s3_file_path)

if prompt_template is None:
    prompt_template = globals.PROMPT_TEMPLATE
    logger.info(f"Using the default local prompt template --> {prompt_template}")
# prompt_template = prompt_template.strip()

# Calculate the number of tokens in the prompt template
# empty_prompt_len_in_tokens = count_tokens(prompt_template.format(context="", question=""))
empty_prompt_len_in_tokens = count_tokens(prompt_template)

# Log the number of tokens
logger.info(f"prompt template length={empty_prompt_len_in_tokens} tokens")

## list all of the files in s3 containing the source data
def list_files():
    response = s3_client.list_objects_v2(Bucket=config['s3_read_data']['read_bucket'], Prefix=config['s3_read_data']['source_data_prefix'])
    return [obj['Key'] for obj in response['Contents']]

# List all files in the bucket and prefix
s3_files = list_files()
logger.info(f"s3 paths of the data set -> {s3_files}")

# Log the files you're going to read
logger.info(f"dataset files = {s3_files}")

# Read and concatenate DataFrames
jsonl_files = [file_key for file_key in s3_files if file_key.endswith('.jsonl')]

# Read and concatenate only the .jsonl files
df = pd.concat([pd.read_json(io.BytesIO(s3_client.get_object(Bucket=config['s3_read_data']['read_bucket'], Key=file_key)['Body'].read()), lines=True) 
                for file_key in jsonl_files])

# Log the source of the dataset and its shape
logger.info(f"dataset read from {s3_files}\nhas shape {df.shape}")


# #### View a portion of the df to view inputs, contexts, and more information on the data
logger.info(f"dataframe with prompts processed --> {df.head()}")

# #### Display basic statistics on the existing dataset: including count, mean, std, min, etc
logger.info(f"distribution of the length field in the dataset is as follows ->\n{df.describe()}")


# ### Concert the dataset elements into prompts as payloads for inference purposes
# Now, we will focus on converting the existing data within our datasets, and extract the information to convert it into prompts to be able to send to our deployed model endpoints during the process of testing and benchmarking for results and various metrics


# Assuming fmbench is a valid Python package and custom prompt preprocess function dir is in it
process_item_dir = Path(pkg_resources.files('fmbench'), 'prompt_preprocess_scripts')
logger.info(f"Using fmbench.prompt_preprocess_scripts directory: {process_item_dir}")

# Ensure the scripts directory exists
process_item_dir.mkdir(parents=True, exist_ok=True)

read_bucket = config['s3_read_data']['read_bucket']
logger.info(f"the read bucket is --> {read_bucket} for reading the preprocess item function files")
process_item_prefix = config['s3_read_data']['prompt_preprocess_dir']
logger.info(f"the preprocess scripts directory is --> {process_item_prefix} for reading the script file names")
process_item_files = config['s3_read_data'].get('preprocess_func_files', [])
logger.info(f"Extracted process_item files that the user has provided --> {process_item_files}")

# Download script files to the fmbench.scripts directory
try:
    for process_item_name in process_item_files:
        # do os.path.join
        s3_script_path = f"{process_item_prefix}/{process_item_name}"
        ## take this out of the loop 
        logger.info(f"the process item path for where the scripts you have entered in s3 will be installed --> {s3_script_path}")
        local_script_path = f"{process_item_dir}/{process_item_name}"
        logger.info(f"Downloading {s3_script_path} to {local_script_path}")
        s3_client.download_file(read_bucket, s3_script_path, str(local_script_path))
except ClientError as error:
    logger.error(f"Failed to download script files: {error}")

# Proceed with getting the function and calling it on df['prompt']
try:
    module_name = Path(config['process_prompt_function']).stem
    logger.info(f"process item function is in file: {module_name}.py")
    process_item_path = Path(process_item_dir) / f"{module_name}.py"
    logger.info(f"process item function path is --> {process_item_path}")

    spec = importlib.util.spec_from_file_location(module_name, str(process_item_path))
    logger.info(f"spec is: {spec}")
    module = importlib.util.module_from_spec(spec)
    logger.info(f"module is: {module}")
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    ## processing the item now
    df['prompt'] = df.apply(lambda row: module.process_item(row, prompt_template), axis=1)
    df['prompt_len'] = df.prompt.map(lambda x: x['prompt_len'])


except Exception as error:  # Broader exception handling for non-ClientError issues
    logger.error(f"An error occurred during deployment: {error}")
    ## processing the item now
    df['prompt'] = df.apply(lambda row: module.process_item(row, prompt_template), axis=1)
    df['prompt_len'] = df.prompt.map(lambda x: x['prompt_len'])


# Convert DataFrame to a CSV format string
csv_buffer = io.StringIO()
df.to_csv(csv_buffer, index=False)
csv_data = csv_buffer.getvalue()
all_prompts_file = config['dir_paths']['all_prompts_file']

# Write to S3 using the write_to_s3 function
write_to_s3(csv_data, config['aws']['bucket'], DATA_DIR, config['dir_paths']['prompts_prefix'], all_prompts_file)

# Log where the prompts are saved
logger.info(f"all prompts dataframe of shape {df.shape} saved to s3://{config['aws']['bucket']}/{DATA_DIR}/{os.path.join(config['dir_paths']['prompts_prefix'], all_prompts_file)}")


## View some of the prompts 
df.head()


# ### Convert Prompts into Payloads for inference purposes
# ------
# Now we will prepare data for model inference. It involves converting prompts, created and stored in a specific format, into payloads for inference. We will utilize the prompt file for our model and incorporate the prompt into a payload using that. 
# 
# These payloads are tailored to the needs of deployed model endpoints. The conversion considers prompt sizes and specific configurations to further make our benchmarking more detailed and comprehensive. 
# 
# The goal is to have a set of well-formatted and parameterized payload requests of various sizes ready to be sent to the model endpoints for inference, with the responses to be used for further analysis


## function to construct payloads (modified for support for both truncate and non truncate inference param)
def construct_request_payload(row, config: Dict) -> Dict:
    
    # Deep copy inference parameters from the config.yml file
    parameters = copy.deepcopy(config['inference_parameters'])
    
    # Check if 'truncate' is present in the inference parameters
    if 'truncate' in parameters:
        if parameters['truncate'] == TRUNCATE_POLICY.AT_PROMPT_TOKEN_LENGTH:
            parameters['truncate'] = row['prompt_len']
    # If 'truncate' is not present, proceed without modifying it
    # Return the constructed payload
    return dict(inputs=row['prompt']['prompt'], parameters=parameters)

def create_dataset_payload_file(df: pd.DataFrame, dataset_info: Dict, config: Dict) -> str:
    # First, log the dataset existing information
    logger.info(f"going to create a payload file as dataset_info={json.dumps(dataset_info, indent=2)}")
    
    # Adjusting filtering to handle the optional 'language' column
    df['prompt_len_in_range'] = df['prompt_len'].apply(lambda x: dataset_info['min_length_in_tokens'] <= x <= dataset_info['max_length_in_tokens'])
    if 'language' in df.columns:
        df_filtered = df[(df['language'] == dataset_info['language']) & df['prompt_len_in_range']]
    else:
        df_filtered = df[df['prompt_len_in_range']]
    
    logger.info(f"after filtering for {json.dumps(dataset_info, indent=2)}, shape of dataframe is {df_filtered.shape}")

    # Construct request payloads for each row in the filtered DataFrame
    df_filtered['request'] = df_filtered.apply(lambda r: construct_request_payload(r, config), axis=1)
    logger.info(f"payload request entry looks like this -> {json.dumps(df_filtered['request'].iloc[0], indent=2)}")
    
    # Convert the 'request' column of the filtered DataFrame to a JSON Lines string
    json_lines_str = df_filtered['request'].to_json(orient='records', lines=True)
    
    # Constructing the file name and S3 path
    lang = dataset_info.get('language', 'all')
    min_len = dataset_info['min_length_in_tokens']
    max_len = dataset_info['max_length_in_tokens']
    file_name = dataset_info['payload_file'].format(lang=lang, min=min_len, max=max_len)

    prompts_path = os.path.join(DATA_DIR, config['dir_paths']['prompts_prefix'])
    s3_file_path = os.path.join(prompts_path, file_name)

    # Write the JSON Lines string to S3
    write_to_s3(json_lines_str, config['aws']['bucket'], DATA_DIR, config['dir_paths']['prompts_prefix'], file_name)

    logger.info(f"dataset of different payload file structures saved to s3://{config['aws']['bucket']}/{s3_file_path}")
    return f"s3://{config['aws']['bucket']}/{s3_file_path}"

items = ((df, d, config) for d in config['datasets'])

# This results in the creation of payload files for each dataset
paths: List = list(itertools.starmap(create_dataset_payload_file, items))

