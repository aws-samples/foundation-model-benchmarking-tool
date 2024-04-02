# Import necessary libraries
import os
import json
import time
import boto3
import logging
import sagemaker
from typing import Dict
from pathlib import Path
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.huggingface import get_huggingface_llm_image_uri

# globals
HF_TOKEN_FNAME: str = os.path.join(os.path.dirname(os.path.realpath(__file__)), "hf_token.txt")

# set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize your S3 client for your model uploading
s3_client = boto3.client('s3')

# Initialize the sagemaker and sagemaker runtime clients 
sm_client = boto3.client("sagemaker")
smr_client = boto3.client("sagemaker-runtime")

# Function to create the llm hugging face model
def create_hugging_face_model(experiment_config: Dict, role_arn: str) -> HuggingFaceModel:
    # Define Model and Endpoint configuration parameter
    model_config = {
    'HF_MODEL_ID': experiment_config['model_id'],
    'SM_NUM_GPUS': json.dumps(experiment_config['env']['NUMBER_OF_GPU']), # Number of GPU used per replica
    'MAX_INPUT_LENGTH': json.dumps(4090),  # Max length of input text
    'MAX_TOTAL_TOKENS': json.dumps(4096),  # Max length of the generation (including input text)
    'MAX_BATCH_TOTAL_TOKENS': json.dumps(8192),  # Limits the number of tokens that can be processed in parallel during the generation
    'HUGGING_FACE_HUB_TOKEN': Path(HF_TOKEN_FNAME).read_text().strip()
    }

    # create HuggingFaceModel with the image uri
    llm_model = HuggingFaceModel(role=role_arn,
                                 image_uri=experiment_config['image_uri'],
                                 env=model_config)

    print(f"Hugging face model defined using {model_config} -> {llm_model}")
    return llm_model

## Function to check the status of the endpoint
def check_endpoint_status(endpoint_name: str) -> str:
    resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
    status = resp["EndpointStatus"]
    while status == "Creating":
        time.sleep(60)
        resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]
    return status

# Deploy the hugging face model
def deploy_hugging_face_model(experiment_config: Dict, llm_model: HuggingFaceModel) -> str:   
    tmout: int = experiment_config['env']['HEALTH_CHECK_TIMEOUT']
    llm = llm_model.deploy(initial_instance_count=experiment_config['env']['INSTANCE_COUNT'],
                           instance_type=experiment_config['instance_type'],
                           container_startup_health_check_timeout=tmout,)
    return llm.endpoint_name

# Function to deploy the model and create the endpoint
def deploy(experiment_config: Dict, role_arn: str) -> Dict[str, str]:
    logger.info("deploying the model using the llm_model and the configurations ....")


    print(f"Setting the model configurations .....")
    llm_model = create_hugging_face_model(experiment_config, role_arn)
    logger.info(f"the llm_model has been defined .... {llm_model}")

    llm_endpoint = deploy_hugging_face_model(experiment_config, llm_model)
    logger.info("Deploying the model now ....")

    status = check_endpoint_status(llm_endpoint)
    logger.info(f"Endpoint status: {status}")

    return dict(endpoint_name=llm_endpoint, experiment_name=experiment_config['name'])