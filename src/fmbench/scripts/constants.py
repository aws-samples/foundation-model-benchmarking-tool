from enum import Enum
from typing import List

CONTAINER_TYPE_DJL: str = 'djl'
CONTAINER_TYPE_VLLM: str = 'vllm'
CONTAINER_TYPE_TRITON: str = 'triton'
TRITON_INFERENCE_SCRIPT: str = '/scripts/triton/triton-vllm-neuronx.sh'
TRITON_CONTENT_DIR_NAME: str = 'triton'
TRITON_SERVE_SCRIPT: str = "triton_serve_model.sh"
AWS_CHIPS_PREFIX_LIST: List[str] = ["inf2", "trn1"]
IS_NEURON_INSTANCE = lambda instance_type: any([instance_type.startswith(p) for p in AWS_CHIPS_PREFIX_LIST])

class ACCELERATOR_TYPE(str, Enum):
    NEURON = 'neuron'
    NVIDIA = "nvidia"

class MODEL_COPIES(str, Enum):
    AUTO = 'auto'
    MAX = "max"
    
    
# These variables represent the platform where a specific
# endpoint is deployed.
PLATFORM_SAGEMAKER: str = "sagemaker"
PLATFORM_EKS: str = "eks"
PLATFORM_EC2: str = "ec2"

# inference server listen port
BASE_PORT_FOR_CONTAINERS: int = 8000
LISTEN_PORT: int = 8080
