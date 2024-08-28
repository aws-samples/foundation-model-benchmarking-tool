# These variables represent the container type that is used
# and play a role in prediction handling for EC2 benchmarking
CONTAINER_TYPE_DJL: str = 'djl'
CONTAINER_TYPE_VLLM: str = 'vllm'

# These variables represent the platform where a specific
# endpoint is deployed.
PLATFORM_SAGEMAKER: str = "sagemaker"
PLATFORM_EKS: str = "eks"
PLATFORM_EC2: str = "ec2"
