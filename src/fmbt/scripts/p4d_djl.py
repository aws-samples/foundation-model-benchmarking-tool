# Import necessary libraries
import os
import time
import boto3
import logging
import tarfile
import sagemaker
from typing import Dict
from pathlib import Path
from urllib.parse import urlparse
from sagemaker.utils import name_from_base
from huggingface_hub import snapshot_download


## set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# globals
HF_TOKEN_FNAME: str = os.path.join(os.path.dirname(os.path.realpath(__file__)), "hf_token.txt")

## Initialize your S3 client for your model uploading
s3_client = boto3.client('s3')

## ------------- Use your specific execution role --------------------------------------------

# role=sagemaker.get_execution_role()  # execution role for the endpoint
sess=sagemaker.session.Session()  # sagemaker session for interacting with different AWS APIs
bucket=sess.default_bucket()  # bucket to house artifacts
model_bucket=sess.default_bucket()  # bucket to house artifacts

## ------------- Define the location of your s3 prefix for model artifacts ----------------------------------------

region=sess._region_name

## Define your account/session id
account_id=sess.account_id()

## ------------- Initialize the sagemaker and sagemaker runtime clients -------------------------------------------

sm_client = boto3.client("sagemaker")
smr_client = boto3.client("sagemaker-runtime")

## ---------------------------------------------------------------------------------------------------------

## Download the model snapshot
def download_model(experiment_config, model_name, local_model_path, allow_patterns):
    
    local_model_path = Path(local_model_path)
    print(f"Local model path: {local_model_path}")
    local_model_path.mkdir(exist_ok=True)
    print(f"Created the local directory: {local_model_path}")

    model_download_path = snapshot_download(
        repo_id=model_name,
        cache_dir=local_model_path,
        allow_patterns=allow_patterns,
        use_auth_token= Path(HF_TOKEN_FNAME).read_text().strip()
    )
    model_artifact = experiment_config['s3_path']
    print(f"Uncompressed model downloaded into ... -> {model_artifact}")
    return model_artifact

## Create the model artifact with the updated serving properties within the directory
def create_and_upload_model_artifact(serving_properties_path, bucket, key):
    # Create a tar.gz file containing only the serving.properties file
    tar_file_path = os.path.join(Path(serving_properties_path).parent, 'model.tar.gz')
    with tarfile.open(tar_file_path, "w:gz") as tar:
        # Add the serving.properties file
        tar.add(serving_properties_path, arcname='serving.properties')

    # Upload the tar.gz file to S3
    model_artifact_path = f"{key}/model.tar.gz"
    s3_client.upload_file(tar_file_path, bucket, model_artifact_path)
    model_tar_gz_path: str = f"s3://{bucket}/{model_artifact_path}"
    logger.info(f"uploaded model.tar.gz to {model_tar_gz_path}")
    return model_tar_gz_path

# Function to create the SageMaker model
def create_model(experiment_config, inference_image_uri, s3_model_artifact, role_arn):
    model_name = name_from_base(experiment_config['model_name'])
    create_model_response = sm_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role_arn,
        PrimaryContainer={
            "Image": inference_image_uri,
            "ModelDataUrl": s3_model_artifact,
            "Environment": experiment_config['env'],
        },
    )
    return model_name, create_model_response["ModelArn"]

## Function to create and deploy the endpoint
def deploy_endpoint(experiment_config, model_name):
    endpoint_config_name = f"{model_name}-config"
    endpoint_name = f"{model_name}-endpoint"

    _ = sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "variant1",
                "ModelName": model_name,
                "InstanceType": experiment_config["instance_type"],
                "InitialInstanceCount": 1,
                "ModelDataDownloadTimeoutInSeconds": 3600,
                "ContainerStartupHealthCheckTimeoutInSeconds": 3600,
            },
        ],
    )
    
    create_endpoint_response = sm_client.create_endpoint(
        EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
    )
    return endpoint_name, create_endpoint_response['EndpointArn']

## Function to check the status of the endpoint
def check_endpoint_status(endpoint_name):
    resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
    status = resp["EndpointStatus"]
    while status == "Creating":
        time.sleep(60)
        resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]
    return status

# Function to deploy the model and create the endpoint
def deploy(experiment_config: Dict, role_arn: str) -> Dict:
    
    model_artifact = experiment_config['s3_path']
    print(f"Uncompressed model downloaded into ... -> {model_artifact}")
    
    logger.info("preparing model artifact...")

    # Set path to serving.properties (update this to the correct path)
    serving_properties_path: str = "p4d_serving.properties"
    dir_path = os.path.dirname(os.path.realpath(__file__))

    properties: str = Path(os.path.join(dir_path, serving_properties_path)).read_text()
    properties = properties.format(s3_path=experiment_config['s3_path'])
    serving_properties_path: str = os.path.join(dir_path, "serving.properties")
    Path(serving_properties_path).write_text(properties)

    # o = urlparse(experiment_config['s3_path'], allow_fragments=False)
    # bucket = o.netloc
    # logger.info(f"bucket name is -> {bucket}")
    # key = o.path
    # logger.info(f"key name is -> {key}")
    bucket = experiment_config['bucket_name']
    logger.info(f"bucket name is -> {bucket}")
    key = experiment_config['key_name']
    logger.info(f"key name is -> {key}")
    logger.info(f"uploading model to S3...bucket={bucket}, key={key}")
    model_artifact = create_and_upload_model_artifact(serving_properties_path, bucket, key)

    logger.info(f"Model uploaded to: {model_artifact}")

    inference_image_uri = experiment_config['image_uri']
    logger.info(f"Inference image URI: {inference_image_uri}")

    model_name, model_arn = create_model(experiment_config, inference_image_uri, model_artifact, role_arn)
    logger.info(f"Created Model: {model_arn}")

    endpoint_name, _ = deploy_endpoint(experiment_config, model_name)
    logger.info(f"Deploying Endpoint: {role_arn}")

    status = check_endpoint_status(endpoint_name)
    logger.info(f"Endpoint status: {status}")

    if status == 'InService':
        logger.info(f"endpoint is in service")
    else:
        logger.info("endpoint is not in service.")

    return dict(endpoint_name=endpoint_name, experiment_name=experiment_config['name'])