import time
from typing import Dict
from sagemaker.predictor import Predictor
from sagemaker.jumpstart.model import JumpStartModel

def deploy(experiment_config: Dict, role_arn: str) -> Dict:
    model = JumpStartModel(
            model_id=experiment_config['model_id'],
            model_version=experiment_config['model_version'],
            image_uri=experiment_config['image_uri'],
            env=experiment_config['env'],
            role=role_arn,
            instance_type=experiment_config['instance_type']
        )

    # Deploy the model using asyncio.to_thread to run in a separate thread
    sec, us = str(time.time()).split(".")
    ep_name = f"{experiment_config['ep_name']}-{sec}-{us}"
    accept_eula = experiment_config.get('accept_eula')
    if accept_eula is not None:
        predictor = model.deploy(initial_instance_count=experiment_config['instance_count'],
                                 accept_eula=accept_eula,
                                 endpoint_name=ep_name)
    else:
        predictor = model.deploy(initial_instance_count=experiment_config['instance_count'],
                                 endpoint_name=ep_name)

    return dict(endpoint_name=predictor.endpoint_name, experiment_name=experiment_config['name'])
