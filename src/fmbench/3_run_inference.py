#!/usr/bin/env python
# coding: utf-8

# ## Run Inference on all deployed endpoints: Various combinations of payloads, concurrency levels, model configurations
# ---------------------
# *This notebook works best with the conda_python3 kernel on a ml.t3.medium machine*.
# 
# #### This step of our solution design includes running inferences on all deployed model endpoints (with different configurations, concurrency levels and payload sizes). This notebook runs inferences in a manner that is calls endpoints concurrently and asychronously to generate responses and record metrics. Here are some of the key components:
# - **Accessing the deployed endpoints**, creating a predictor object for these endpoints to call them during inference time.
# - **Functions to define metrics**: This notebook sets stage for metrics to be recorded during the time of invocation of all these models for benchmarking purposes.
# - **Running Actual Inferences**: Once the metrics are defined, we set a blocker function that is responsible for creating inference on a single payload called get_inference. We then run a series of asynchronous functions that can be viewed in the code (link above), to create asychronous inferefences on the deployed models. The way we send requests are by creating combinations: this means creating combinations of payloads of different sizes that can be viewed in the config.yml file, with different concurrency levels (in this case we first go through all patches of payloads with a concurrency level of 1, then 2, and then 4). You can set this to your desired value.


# ### Import all of the necessary libraries below to run this notebook
import io
import sys
import glob
import time
import json
import copy
import boto3
import asyncio
import logging
import itertools
import sagemaker
import pandas as pd
import importlib.util
from fmbench.utils import *
from fmbench.globals import * ## add only the vars needed import globals as g.
from datetime import datetime
from datetime import timezone
from transformers import AutoTokenizer
from sagemaker.predictor import Predictor
import importlib.resources as pkg_resources
from botocore.exceptions import ClientError
from sagemaker.serializers import JSONSerializer
from typing import Dict, List, Optional, Tuple, Union


## set a logger to track all logs during the run of this inference step
logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


#### Load the Config.yml file that contains information that is used across this benchmarking environment
config = load_config(CONFIG_FILE)
logger.info(json.dumps(config, indent=2))

## set the date time of the run of this inference step
date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

## getting access to the s3 bucket where endpoints.json for different models resides
s3_client = boto3.client('s3')


# ### Access the deployed model endpoints from the endpoints.json file 
## Refer to the file path for the endpoint
## getting the endpoint as an s3 object from the deployed path
endpoint_info_list = json.loads(get_s3_object(config['aws']['bucket'], ENDPOINT_LIST_PATH))
logger.info(f"found information for {len(endpoint_info_list)} endpoints in bucket={config['aws']['bucket']}, key={ENDPOINT_LIST_PATH}")
logger.info(json.dumps(endpoint_info_list, indent=2))

# List down the endpoint names that have been deployed
endpoint_name_list = [e['endpoint']['EndpointName'] for e in endpoint_info_list]
logger.info(f"there are {len(endpoint_name_list)} deployed endpoint(s), endpoint_name_list->{endpoint_name_list}")


# ### Creating predictor objects from the deployed endpoints
# create predictor objects

## create a sagemaker predictor for these endpoints
def create_predictor(endpoint_name: str) -> Optional[sagemaker.base_predictor.Predictor]:
    # Create a SageMaker Predictor object
    predictor = Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=sagemaker.Session(),
        serializer=JSONSerializer()
    )
    return predictor

## Display the list of predictor objects that have been deployed ready for inferencing from
predictor_list: List = [create_predictor(ep) for ep in endpoint_name_list]
logger.info(predictor_list)


# ### Creating functions to define and calculate metrics during the time of invocations
def safe_sum(l: List) -> Union[int, float]:
    return sum(filter(None, l))

def safe_div(n: Union[int, float], d: Union[int, float]) -> Optional[Union[int, float]]:
    return n/d if d else None

## Represents the function to calculate all of the metrics at the time of inference
def calculate_metrics(responses, chunk, elapsed_async, experiment_name, concurrency, payload_file) -> Dict:
    
    ## calculate errors based on the completion status of the inference prompt
    errors = [r for r in responses if r['completion'] is None]
    
    ## Calculate the difference as the successes 
    successes = len(chunk) - len(errors)
    
    ## Count all of the prompts token count during inference
    all_prompts_token_count = safe_sum([r['prompt_tokens'] for r in responses])
    prompt_token_throughput = round(all_prompts_token_count / elapsed_async, 2)
    prompt_token_count_mean = safe_div(all_prompts_token_count, successes)
    all_completions_token_count = safe_sum([r['completion_tokens'] for r in responses])
    completion_token_throughput = round(all_completions_token_count / elapsed_async, 2)
    completion_token_count_mean = safe_div(all_completions_token_count, successes)
    transactions_per_second = round(successes / elapsed_async, 2)
    transactions_per_minute = int(transactions_per_second * 60)
    
    ## calculate the latency mean utilizing the safe_sum function defined above
    latency_mean = safe_div(safe_sum([r['latency'] for r in responses]), successes)
    
    ## Function returns all these values at the time of the invocations
    return {
        'experiment_name': experiment_name,
        'concurrency': concurrency,
        'payload_file': payload_file,
        'errors': errors,
        'successes': successes,
        'error_rate': len(errors)/len(chunk),
        'all_prompts_token_count': all_prompts_token_count,
        'prompt_token_count_mean': prompt_token_count_mean,
        'prompt_token_throughput': prompt_token_throughput,
        'all_completions_token_count': all_completions_token_count,
        'completion_token_count_mean': completion_token_count_mean,
        'completion_token_throughput': completion_token_throughput,
        'transactions': len(chunk),
        'transactions_per_second': transactions_per_second,
        'transactions_per_minute': transactions_per_minute,
        'latency_mean': latency_mean
    }


# ### Set a blocker function and a series of asynchronous concurrent model prompt invocations
## this function (or a similar function that returns a dict) will be in the inference file so that inference is recorded during the ocurse of the prediction
def set_metrics(endpoint_name=None,
                    prompt=None,
                    inference_params=None,
                    completion=None,
                    prompt_tokens=None,
                    completion_tokens=None,
                    latency=None) -> Dict:
    return dict(endpoint_name=endpoint_name,                
                prompt=prompt,
                **inference_params,
                completion=completion,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency=latency)

def get_model_inference(experiment_config: Dict, predictor, payload) -> Dict:
    
    smr_client = boto3.client("sagemaker-runtime")

    # Assuming fmbench is a valid Python package and scripts is a subdirectory within it
    inference_scripts_dir = Path(pkg_resources.files('fmbench'), 'inference_scripts')
    logger.info(f"Using the inference scripts directory directory: {inference_scripts_dir}")

    # Ensure the inference scripts directory exists
    inference_scripts_dir.mkdir(parents=True, exist_ok=True)

    read_bucket = config['s3_read_data']['read_bucket']
    logger.info(f"the read bucket is --> {read_bucket} for reading the script files")
    inf_scripts_prefix = config['s3_read_data']['inf_scripts_prefix']
    logger.info(f"the inference scripts directory is --> {inf_scripts_prefix} for reading the script file names")
    inf_script_files = config['s3_read_data'].get('inf_scripts_prefix', [])
    logger.info(f"Extracted inference script files that the user has provided --> {inf_script_files}")

    # Download script files to the fmbench.inference_scripts directory
    try:
        for inf_script_name in inf_script_files:
            # do os.path.join
            s3_inf_script_path = f"{inf_scripts_prefix}/{inf_script_name}"
            ## take this out of the loop 
            logger.info(f"the inference script path for where the scripts you have entered in s3 will be installed --> {s3_inf_script_path}")
            local_script_path = inference_scripts_dir / inf_script_name
            logger.info(f"Downloading {s3_inf_script_path} to {local_script_path}")
            s3_client.download_file(read_bucket, s3_inf_script_path, str(local_script_path))
    except ClientError as error:
        logger.error(f"Failed to download inference script files: {error}")

    # Proceed with inference
    try:
        module_name = Path(experiment_config['inference_script']).stem
        logger.info(f"The given script provided for inference of this model is --> {module_name}")
        inference_script_path = inference_scripts_dir / f"{module_name}.py"
        logger.info(f"Inference script path is --> {inference_script_path}")

        # Check and proceed with local script
        if not inference_script_path.exists():
            logger.error(f"Inference script {inference_script_path} not found.")
            return None

        logger.info(f"Inference process execution using local code: {inference_script_path}")

        spec = importlib.util.spec_from_file_location(module_name, str(inference_script_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)


        ## this is a custom get inference function that will be used from the custom inference file you get if your models support different inference payloads than the one fmbench supports out of the box
        inference_result = module.get_inference(predictor, payload)
        return inference_result

    except Exception as error:  # Broader exception handling for non-ClientError issues
        logger.error(f"An error occurred during inference: {error}")
        return inference_result


# ### Setting a series of asynchronous functions to invoke and run inferences concurrently and asynchronously

## Represents a function to start invoking models in separate thread asynchronously for the blocker function
async def async_get_inference(experiment_config: Dict, predictor, payload: Dict) -> Dict:
    return await asyncio.to_thread(get_model_inference, experiment_config, predictor, payload)

## Gathers all of the tasks and sets of the concurrent calling of the asychronous invocations
async def async_get_all_inferences(config: Dict, predictor, payload_list: List) -> List:
    return await asyncio.gather(*[async_get_inference(config, predictor, payload) for payload in payload_list])

## This function runs the asynchronous function series above together for different experiments and concurrency levels.
async def run_inferences(predictor: sagemaker.base_predictor.Predictor, chunk: List, experiment: Dict, concurrency: int, payload_file: str) -> Tuple[List, Dict]:
    logger.info(f"Processing chunk with concurrency={concurrency}")
    s = time.perf_counter()
    responses = await async_get_all_inferences(experiment, predictor, chunk)
    elapsed_async = time.perf_counter() - s

    # Add more metadata about this experiment
    for r in responses:
        r['experiment_name'] = experiment['name']
        r['concurrency'] = concurrency

    metrics = calculate_metrics(responses, chunk, elapsed_async, experiment['name'], concurrency, payload_file)
    return responses, metrics

## Function to create the predictors from the experiment we are iterating over
def create_predictor_for_experiment(experiment: str, config: Dict, endpoint_info_list: List) -> Optional[sagemaker.base_predictor.Predictor]:

    ## Here, we set the index and then iterate through the experiments
    e_idx = config['experiments'].index(experiment) + 1

    ## Iterate through the endpoint information to fetch the endpoint name
    ep_info = [e for e in endpoint_info_list if e['experiment_name'] == experiment['name']]
    if not ep_info:
        logger.error(f"endpoint for experiment={experiment['name']} not found, skipping")
        return None
    ep_name = ep_info[0]['endpoint']['EndpointName']
    logger.info(f"experiment={e_idx}, name={experiment['name']}, ep_name={ep_name}")

    # create a predictor from each endpoint in experiments
    return create_predictor(ep_name)

## Here, we will process combinations of concurrency levels, the payload files and then loop through the 
## different combinations to make payloads splitted in terms of the concurrency metric and how we can run 
## it and make inference

def create_payload_dict(jline: str, experiment: Dict) -> Dict:
    payload: Dict = json.loads(jline)
    if experiment.get('remove_truncate', False) is True:
        if payload['parameters'].get('truncate'):
            del payload['parameters']['truncate']
    return payload
    
    
def create_combinations(experiment: Dict) -> List[Tuple]:
    combinations_data = []

    # Repeat for each concurrency level
    combinations = list(itertools.product(experiment['concurrency_levels'], experiment['payload_files']))
    logger.info(f"there are {len(combinations)} combinations of {combinations} to run")

    for concurrency, payload_file in combinations:
        # Construct the full S3 file path
        s3_file_path = os.path.join(PROMPTS_DIR, payload_file)
        logger.info(f"s3 path where the payload files are being read from -> {s3_file_path}")

        # Read the payload file from S3
        try:
            response = s3_client.get_object(Bucket=config['aws']['bucket'], Key=s3_file_path)
            payload_file_content = response['Body'].read().decode('utf-8')

            # Create a payload list by processing each line
            payload_list = [create_payload_dict(jline, experiment) for jline in payload_file_content.splitlines()]
            logger.info(f"read from s3://{config['aws']['bucket']}/{s3_file_path}, contains {len(payload_list)} lines")

        except Exception as e:
            logger.error(f"Error reading file from S3: {e}")
            continue

        logger.info(f"creating combinations for concurrency={concurrency}, payload_file={payload_file}, payload_list length={len(payload_list)}")
        
        n = concurrency
        
        if len(payload_list) < n:
            elements_to_add = n - len(payload_list)
            element_to_replicate = payload_list[0]
            # payload_list = payload_list.extend([element_to_replicate]*elements_to_add)
            payload_list.extend([element_to_replicate]*elements_to_add)
            
        # Split the original list into sublists which contain the number of requests we want to send concurrently        
        payload_list_splitted = [payload_list[i * n:(i + 1) * n] for i in range((len(payload_list) + n - 1) // n )]  
        
        for p in payload_list_splitted:
            if len(p) < n:
                elements_to_add = n - len(p)
                element_to_replicate = p[0]
                # p = p.extend([element_to_replicate]*elements_to_add)
                p.extend([element_to_replicate]*elements_to_add)
            

        # Only keep lists that have at least concurrency number of elements
        len_before = len(payload_list_splitted)
        payload_list_splitted = [p for p in payload_list_splitted if len(p) == concurrency]
        logger.info(f"after only retaining chunks of length {concurrency}, we have {len(payload_list_splitted)} chunks, previously we had {len_before} chunks")
        combinations_data.append((concurrency, payload_file, payload_list_splitted))
    logger.info(f"there are {len(combinations)} for {experiment}")
    return combinations_data

# for each experiment
#   - for each endpoint and concurrency in an experiment

def clear_dir(dir_path: str):
    files = glob.glob(os.path.join(dir_path, "*"))
    for f in files:
        os.remove(f)


_ = list(map(clear_dir, [METRICS_PER_INFERENCE_DIR, METRICS_PER_CHUNK_DIR]))

## Initializing the total model instance cost to 0
total_model_instance_cost: int = 0

## To keep track of the cost for all model endpoints
cost_data = []

## To keep track of the experiment durations and the time it takes for the model endpoint to be in service to calculate cost association
experiment_durations = []  

## start the timer before the start of inferences
current_time = datetime.now(timezone.utc)
logger.info(f"Current time recorded while running this experiment is {current_time}..... deployed models are going to start inferences...")

num_experiments: int = len(config['experiments'])
for e_idx, experiment in enumerate(config['experiments']):
    e_idx += 1  # Increment experiment index
    experiment_start_time = time.perf_counter()  # Start timer for the experiment

    predictor = create_predictor_for_experiment(experiment, config, endpoint_info_list)
    if predictor is None:
        logger.error(f"predictor could not be created for experiment={experiment}, moving to next...")
        continue

    combination_data = create_combinations(experiment)

    for concurrency, payload_file, split_payload in combination_data:
        for chunk_index, chunk in enumerate(split_payload):
            logger.info(f"e_idx={e_idx}/{num_experiments}, chunk_index={chunk_index+1}/{len(split_payload)}")

            # responses, metrics = await run_inferences(predictor, chunk, experiment, concurrency, payload_file)
            responses, metrics = asyncio.run(run_inferences(predictor, chunk, experiment, concurrency, payload_file))

            if metrics:
                metrics_json = json.dumps(metrics, indent=2)
                metrics_file_name = f"{time.time()}.json"
                metrics_s3_path = os.path.join(METRICS_PER_CHUNK_DIR, metrics_file_name)
                write_to_s3(metrics_json, config['aws']['bucket'], "", METRICS_PER_CHUNK_DIR, metrics_file_name)

            if responses:
                for r in responses:
                    response_json = json.dumps(r, indent=2)
                    response_file_name = f"{time.time()}.json"
                    response_s3_path = os.path.join(METRICS_PER_INFERENCE_DIR, response_file_name)
                    write_to_s3(response_json, config['aws']['bucket'], "", METRICS_PER_INFERENCE_DIR, response_file_name)
    
    ## initializing the experiment cost
    exp_cost = 0
    
    # Experiment done, stopping the timer for this given experiment
    experiment_end_time = time.perf_counter()

    # calculating the duration of this given endpoint inference time
    experiment_duration = experiment_end_time - experiment_start_time
    logger.info(f"the {experiment['name']} ran for {experiment_duration} seconds......")

    # calculating the per second cost for this instance type
    exp_instance_type: str = experiment['instance_type']

    # price of the given instance for this experiment 
    hourly_rate = config['pricing'].get(experiment['instance_type'], 0)
    logger.info(f"the hourly rate for {experiment['name']} running on {exp_instance_type} is {hourly_rate}")

    cost_per_second = hourly_rate / 3600
    logger.info(f"the rate for {experiment['name']} running on {exp_instance_type} is {cost_per_second} per second")
    
    #cost for this given exp
    exp_cost = experiment_duration * cost_per_second
    logger.info(f"the rate for running {experiment['name']} running on {exp_instance_type} for {experiment_duration} is ${exp_cost}....")

    ## tracking the total cost
    total_model_instance_cost += exp_cost

    experiment_durations.append({
        'experiment_name': experiment['name'],
        'instance_type': exp_instance_type, 
        'duration_in_seconds': f"{experiment_duration:.2f}", 
        'cost': f"{exp_cost:.2f}", 
    })

    logger.info(f"experiment={e_idx}/{num_experiments}, name={experiment['name']}, duration={experiment_duration:.2f} seconds, done")

# experiment_durations.append({'total_cost': f"${total_model_instance_cost:.2f}"})

# After all experiments are done, summarize and optionally save experiment durations along with costs
df_durations = pd.DataFrame(experiment_durations)
logger.info(f"experiment durations: {df_durations}")

# Convert the DataFrame to CSV and write it to S3 or wherever you prefer
csv_buffer_cost = io.StringIO()
df_durations.to_csv(csv_buffer_cost, index=False)
experiment_associated_cost = csv_buffer_cost.getvalue()

# Assuming write_to_s3() is already defined and configured correctly
write_to_s3(experiment_associated_cost, config['aws']['bucket'], "", METRICS_DIR, SUMMARY_MODEL_ENDPOINT_COST_PER_INSTANCE)
logger.info(f"Summary for cost of instance per endpoint per run saved to s3://{config['aws']['bucket']}/{METRICS_DIR}/{SUMMARY_MODEL_ENDPOINT_COST_PER_INSTANCE}")

logger.info(f"total cost of all experiments: ${df_durations.cost.sum()}")

# List .json files in the specified S3 directory
s3_files = list_s3_files(config['aws']['bucket'], METRICS_PER_INFERENCE_DIR)

# Read and parse each JSON file from S3
json_list = list(map(lambda key: json.loads(get_s3_object(config['aws']['bucket'], key)), \
                     s3_files))

# Create DataFrame
df_responses = pd.DataFrame(json_list)
logger.info(f"created dataframe of shape {df_responses.shape} from all responses")
df_responses.head()

# List .json files in the specified S3 directory
s3_files = list_s3_files(config['aws']['bucket'], METRICS_PER_CHUNK_DIR)

# Read and parse each JSON file from S3
json_list = list(map(lambda key: json.loads(get_s3_object(config['aws']['bucket'], key)), \
                     s3_files))

# Create DataFrame
df_metrics = pd.DataFrame(json_list)
logger.info(f"created dataframe of shape {df_metrics.shape} from all responses")
df_metrics.head()

df_endpoints = pd.json_normalize(endpoint_info_list)
df_endpoints['instance_type'] = df_endpoints['endpoint_config.ProductionVariants'].map(lambda x: x[0]['InstanceType'])
df_endpoints
cols_for_env = [c for c in df_endpoints.columns if 'Environment' in c]
print(cols_for_env)
cols_of_interest = ['experiment_name', 
                    'instance_type',
                    'endpoint.EndpointName',
                    'model_config.ModelName',
                    'model_config.PrimaryContainer.Image',   
                    'model_config.PrimaryContainer.ModelDataSource.S3DataSource.S3Uri']
cols_of_interest.extend(cols_for_env)

df_endpoints = df_endpoints[cols_of_interest]
df_endpoints = df_endpoints[cols_of_interest]
cols_of_interest_renamed = [c.split('.')[-1] for c in cols_of_interest]
df_endpoints.columns = cols_of_interest_renamed

# Check if 'experiment_name' column exists in both DataFrames
print("Columns in df_responses:", df_responses.columns)
print("Columns in df_endpoints:", df_endpoints.columns)

# Merge operation
df_results = pd.merge(left=df_responses, right=df_endpoints, how='left', left_on='experiment_name', right_on='experiment_name')

# Inspect the result
df_results.head()

df_results = pd.merge(left=df_responses, right=df_endpoints, how='left', left_on='experiment_name', right_on='experiment_name')
df_results.head()

# Convert df_results to CSV and write to S3
csv_buffer = io.StringIO()
df_results.to_csv(csv_buffer, index=False)
csv_data_results = csv_buffer.getvalue()
results_file_name = config['results']['per_inference_request_file'].format(datetime=date_time)
results_s3_path = os.path.join(METRICS_DIR, results_file_name)
logger.info(f"results s3 path for per inference csv --> {results_s3_path}")
write_to_s3(csv_data_results, config['aws']['bucket'], "", METRICS_DIR, results_file_name)
logger.info(f"saved results dataframe of shape={df_results.shape} in s3://{BUCKET_NAME}/{results_s3_path}")

# Ensure the metadata directory exists
os.makedirs(METADATA_DIR, exist_ok=True)

# Path for the metrics_path.txt file
metrics_path_file = os.path.join(METADATA_DIR, 'metrics_path.txt')
logger.info(f"the metrics metadata path is saved here --> {metrics_path_file}")

# Write the METRICS_DIR to metrics_path.txt
with open(metrics_path_file, 'w') as file:
    file.write(METRICS_DIR)

## Write this data to S3
write_to_s3(METRICS_DIR, config['aws']['bucket'], "", DATA_DIR, 'metrics_path.txt')

logger.info(f"the information on the defined path for results on these metrics are given in this --> {METRICS_DIR}")

logger.info(f"df_metrics cols = {df_metrics.columns}")
logger.info(f"df_endpoints cols = {df_endpoints.columns}")
df_metrics = pd.merge(left=df_metrics, right=df_endpoints, how='left', left_on='experiment_name', right_on='experiment_name')
df_metrics.head()

# Convert df_metrics to CSV and write to S3
csv_buffer = io.StringIO()
df_metrics.to_csv(csv_buffer, index=False)
csv_data_metrics = csv_buffer.getvalue()
metrics_file_name = config['results']['all_metrics_file'].format(datetime=date_time)
metrics_s3_path = os.path.join(METRICS_DIR, metrics_file_name)
logger.info(f"results s3 path for metrics csv --> {metrics_s3_path}")
write_to_s3(csv_data_metrics, config['aws']['bucket'], "", METRICS_DIR, metrics_file_name)
logger.info(f"saved metrics results dataframe of shape={df_metrics.shape} in s3://{config['aws']['bucket']}/{metrics_s3_path}")

