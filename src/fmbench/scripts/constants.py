from enum import Enum
from typing import List

CONTAINER_TYPE_DJL: str = 'djl'
CONTAINER_TYPE_VLLM: str = 'vllm'
CONTAINER_TYPE_TRITON: str = 'triton'
CONTAINER_TYPE_OLLAMA: str = 'ollama'
CONTAINER_TYPE_HUGGINGFACE: str = 'huggingface'
TRITON_INFERENCE_SCRIPT_VLLM: str = '/scripts/triton/triton-vllm-neuronx.sh'
TRITON_INFERENCE_SCRIPT_DJL: str = '/scripts/triton/triton-djl-python-neuronx.sh'
TRITON_CONTENT_DIR_NAME_VLLM: str = 'triton/vllm'
TRITON_CONTENT_DIR_NAME_DJL: str = 'triton/djl'
TRITON_SERVE_SCRIPT: str = "triton_serve_model.sh"
AWS_CHIPS_PREFIX_LIST: List[str] = ["inf2", "trn1"]
IS_NEURON_INSTANCE = lambda instance_type: any([instance_type.startswith(p) for p in AWS_CHIPS_PREFIX_LIST])

class ACCELERATOR_TYPE(str, Enum):
    NEURON = 'neuron'
    NVIDIA = "nvidia"

class BACKEND(str, Enum):
    VLLM_BACKEND = 'vllm'
    DJL_BACKEND = 'djl'
    TENSORRT_BACKEND = 'tensorrt'

class MODEL_COPIES(str, Enum):
    AUTO = 'auto'
    MAX = "max"
    
    
# These variables represent the platform where a specific
# endpoint is deployed.
PLATFORM_SAGEMAKER: str = "sagemaker"
PLATFORM_EKS: str = "eks"
PLATFORM_EC2: str = "ec2"
PLATFORM_BEDROCK: str = "bedrock"
PLATFORM_EXTERNAL: str = "external"

# inference server listen port
BASE_PORT_FOR_CONTAINERS: int = 8000
LISTEN_PORT: int = 8080

# This is the file where the EC2 instance utilization metrics are stored
EC2_SYSTEM_METRICS_FNAME: str = "EC2_system_metrics.csv"
EC2_UTILIZATION_METRICS_INTERVAL: int = 5
