from typing import List

CONTAINER_TYPE_DJL: str = 'djl'
CONTAINER_TYPE_VLLM: str = 'vllm'

AWS_CHIPS_PREFIX_LIST: List[str] = ["inf2", "trn1"]
IS_NEURON_INSTANCE = lambda instance_type: any([instance_type.startswith(p) for p in AWS_CHIPS_PREFIX_LIST])
