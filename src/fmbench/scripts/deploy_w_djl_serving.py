"""
Deploys a model from HuggingFace on Amazon SageMaker using the DJL DeepSpeek LMI
(https://github.com/deepjavalibrary/djl-serving).

1. Configuration is read from the configured serving.properties file.
2. A hf_token.txt file is required to download the model from Hugging Face.
"""
# Import necessary libraries
import os
import glob
import time
import boto3
import logging
import tarfile
import tempfile
import sagemaker
from pathlib import Path
from urllib.parse import urlparse
from sagemaker.utils import name_from_base
from huggingface_hub import snapshot_download
from typing import Dict, List, Tuple, Optional


# set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# globals
HF_TOKEN_FNAME: str = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   "hf_token.txt")

# Initialize your S3 client for your model uploading
s3_client = boto3.client('s3')

# session/account specific variables
sess = sagemaker.session.Session()
# Define the location of your s3 prefix for model artifacts
region: str =sess._region_name

# bucket to house model artifacts
default_bucket = sess.default_bucket()

# Define your account id
account_id: str = sess.account_id()

# Initialize the sagemaker and sagemaker runtime clients
sm_client = boto3.client("sagemaker")

def _download_model(model_id: str,
                    local_model_path: str,
                    allow_patterns: Optional[List] = ["*"]) -> str:
    """
    Download the model files locally
    """
    local_model_path = Path(local_model_path)
    print(f"Local model path: {local_model_path}")
    local_model_path.mkdir(exist_ok=True)
    print(f"Created the local directory: {local_model_path}")

    model_download_path = snapshot_download(
        repo_id=model_id,
        cache_dir=local_model_path,
        allow_patterns=allow_patterns,
        use_auth_token=Path(HF_TOKEN_FNAME).read_text().strip()
    )
    print(f"Uncompressed model downloaded into ... -> {model_download_path}")
    return model_download_path

def _upload_dir(localDir: str, awsInitDir: str, bucketName: str, tag: str ="*.*"):
    s3 = boto3.resource('s3')
    p = Path(localDir)
    # Iterate over all directories and files within localDir
    for path in p.glob('**/*'):
        if path.is_file():
            rel_path = path.relative_to(p)
            awsPath = os.path.join(awsInitDir, str(rel_path)).replace("\\", "/")
            logger.info(f"Uploading {path} to s3://{bucketName}/{awsPath}")
            logger.info(f"path: {path}, bucket name: {bucketName}, awsPath: {awsPath}")
            s3.meta.client.upload_file(path, bucketName, awsPath)

def _create_and_upload_model_artifact(serving_properties_path: str,
                                      bucket: str,
                                      prefix: str) -> str:
    """
    Create the model artifact with the updated serving properties within the directory
    """
    # Create a tar.gz file containing only the serving.properties file
    tar_file_path = os.path.join(Path(serving_properties_path).parent, 'model.tar.gz')
    with tarfile.open(tar_file_path, "w:gz") as tar:
        # Add the serving.properties file
        tar.add(serving_properties_path, arcname='serving.properties')

    # Upload the tar.gz file to S3
    key = f"{prefix}/model.tar.gz"
    s3_client.upload_file(tar_file_path, bucket, key)
    model_tar_gz_path: str = f"s3://{bucket}/{key}"
    logger.info(f"uploaded model.tar.gz to {model_tar_gz_path}")
    return model_tar_gz_path

def _create_model(experiment_config: Dict,
                  inference_image_uri: str,
                  s3_model_artifact: str,
                  role_arn: str) -> Tuple[str, str]:
    """
    # Function to create the SageMaker model
    """
    model_name = name_from_base(experiment_config['model_name'])
    env = experiment_config.get('env')
    if env:
        pc = dict(Image=inference_image_uri,
                  ModelDataUrl=s3_model_artifact,
                  Environment=env)
    else:
        pc = dict(Image=inference_image_uri,
                  ModelDataUrl=s3_model_artifact)
    create_model_response = sm_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role_arn,
        PrimaryContainer=pc,
    )
    return model_name, create_model_response["ModelArn"]


def _deploy_endpoint(experiment_config: Dict,
                     model_name: str) -> Tuple[str, str]:
    """
    Function to create and deploy the endpoint
    """
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


def _check_endpoint_status(endpoint_name: str) -> str:
    """
    Function to check the status of the endpoint
    """
    resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
    status = resp["EndpointStatus"]
    while status == "Creating":
        time.sleep(60)
        resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]
    return status


def deploy(experiment_config: Dict, role_arn: str) -> Dict:
    """
    Function to deploy the model and create the endpoint
    """

    if experiment_config.get("download_from_hf_place_in_s3") is True:
        with tempfile.TemporaryDirectory() as local_model_path:
            logger.info(f"created temporary directory {local_model_path}")
        local_model_path = _download_model(experiment_config['model_id'],
                                           local_model_path)
        logger.info(f"going to upload model files to {experiment_config['model_s3_path']}")

        o = urlparse(experiment_config['model_s3_path'], allow_fragments=False)
        _upload_dir(local_model_path, o.path.lstrip('/'), o.netloc) 
        logger.info(f"local model path: {local_model_path}, o.path: {o.path}, o.netloc: {o.netloc}")

        model_artifact = experiment_config['model_s3_path']
        logger.info(f"Uncompressed model downloaded into ... -> {model_artifact}")

    logger.info("preparing model artifact...")

    # handle serving.properties, we read it from the config and then write it to
    # a local file
    write_bucket = experiment_config.get('bucket', default_bucket)
    logger.info(f"write bucket for inserting model.tar.gz into: {write_bucket}")
    properties = experiment_config["serving.properties"].format(write_bucket=write_bucket)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    serving_properties_path = os.path.join(dir_path, "serving.properties")
    Path(serving_properties_path).write_text(properties)
    logger.info(f"written the following serving.properties \
                  content={properties} to {serving_properties_path}")

    # create and upload the model.tar.gz, note that this file is just a placeholder
    # it is not the actual model, the actual model binaries are in s3 or HuggingFace
    # and the container will download them when the model endpoint is being created
    logger.info(f"uploading model.tar.gz to S3,bucket={write_bucket}, \
                  prefix={experiment_config['model_id']}")
    model_artifact = _create_and_upload_model_artifact(serving_properties_path,
                                                      write_bucket,
                                                      experiment_config['model_id'])
    logger.info(f"model uploaded to: {model_artifact}")

    inference_image_uri = experiment_config['image_uri']
    logger.info(f"inference image URI: {inference_image_uri}")

    # create model
    model_name, model_arn = _create_model(experiment_config,
                                          inference_image_uri,
                                          model_artifact,
                                          role_arn)
    logger.info(f"created Model: {model_arn}")

    # deploy model
    endpoint_name, _ = _deploy_endpoint(experiment_config, model_name)
    logger.info(f"deploying endpoint: {endpoint_name}")

    # check model deployment status
    status = _check_endpoint_status(endpoint_name)
    logger.info(f"Endpoint status: {status}")

    if status == 'InService':
        logger.info("endpoint is in service")
    else:
        logger.info("endpoint is not in service.")

    return dict(endpoint_name=endpoint_name,
                experiment_name=experiment_config['name'])
