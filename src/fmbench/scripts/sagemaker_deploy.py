import boto3
import fmbench
import logging
import sagemaker
from typing import Dict
from sagemaker import Model
from fmbench.scripts import constants
from sagemaker.utils import name_from_base

logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize the platform where this script deploys the model
PLATFORM: str = constants.PLATFORM_SAGEMAKER

tag = [
    {
        'Key': 'fmbench-version',
        'Value': fmbench.__version__
    }
]


def deploy(experiment_config: Dict, role_arn: str) -> Dict:
    role = role_arn
    endpoint_name = name_from_base(experiment_config['ep_name'])

    boto3_session=boto3.session.Session()
    smr = boto3.client('sagemaker-runtime')
    sm = boto3.client('sagemaker')
    # sagemaker session for interacting with different AWS APIs
    sess = sagemaker.session.Session(boto3_session, 
                                     sagemaker_client=sm, 
                                     sagemaker_runtime_client=smr)
    logger.info("======================================")
    logger.info(f"Will load artifacts from {experiment_config['model_data']}")
    logger.info("======================================")

    logger.info("======================================")
    logger.info(f"Using Container image {experiment_config['image_uri']}")
    logger.info("======================================")
    model = Model(name=endpoint_name,
                  # Enable SageMaker uncompressed model artifacts
                  model_data=experiment_config['model_data'],
                  image_uri=experiment_config['image_uri'],
                  role=role,
                  env=experiment_config['env'],
                  sagemaker_session=sess)
    logger.info(model)

    logger.info(f'\nModel deployment initiated Endpoint Name: {endpoint_name}\n')
    if experiment_config.get('accept_eula') is not None:
        model.deploy(
            initial_instance_count=experiment_config['instance_count'],
            instance_type=experiment_config['instance_type'],
            endpoint_name=endpoint_name,
            #volume_size=512, # not allowed for the selected Instance type ml.g5.12xlarge
            model_data_download_timeout=1200, # increase the timeout to download large model
            container_startup_health_check_timeout=1200, # increase the timeout to load large model,
            wait=True,
            tags=tag, 
            accept_eula=experiment_config.get('accept_eula')
        )
    else:
        model.deploy(
            initial_instance_count=experiment_config['instance_count'],
            instance_type=experiment_config['instance_type'],
            endpoint_name=endpoint_name,
            #volume_size=512, # not allowed for the selected Instance type ml.g5.12xlarge
            model_data_download_timeout=1200, # increase the timeout to download large model
            container_startup_health_check_timeout=1200, # increase the timeout to load large model,
            wait=True,
            tags=tag 
        )
    logger.info(f'Model deployment on Endpoint Name: {endpoint_name} finished\n')
    return dict(endpoint_name=endpoint_name,
                experiment_name=experiment_config['name'],
                instance_type=experiment_config['instance_type'],
                instance_count=experiment_config['instance_count'], 
                deployed=True)
