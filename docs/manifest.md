# Config files

The following listing provides names and paths of all the configuration files currently available out of the box with `FMBench`. You can find these files in the [configs](https://github.com/aws-samples/foundation-model-benchmarking-tool/tree/main/src/fmbench/configs) directory in the `FMBench` GitHub repo. Use these files as=is or to create your own custom configuration files.

``` bash

configs
├── bedrock
│   ├── config-bedrock-claude.yml
│   ├── config-bedrock-llama3-streaming.yml
│   ├── config-bedrock-llama3.yml
│   ├── config-bedrock-titan-text-express.yml
│   └── config-bedrock.yml
├── bert
│   └── config-distilbert-base-uncased.yml
├── byoe
│   └── config-model-byo-sagemaker-endpoint.yml
├── eks_manifests
│   ├── llama3-ray-service.yaml
│   └── mistral-ray-service.yaml
├── gemma
│   └── config-gemma-2b-g5.yml
├── llama2
│   ├── 13b
│   │   ├── config-bedrock-sagemaker-llama2.yml
│   │   ├── config-byo-rest-ep-llama2-13b.yml
│   │   ├── config-llama2-13b-inf2-g5-p4d.yml
│   │   └── config-llama2-13b-inf2-g5.yml
│   ├── 70b
│   │   ├── config-ec2-llama2-70b.yml
│   │   ├── config-llama2-70b-g5-p4d-tgi.yml
│   │   ├── config-llama2-70b-g5-p4d-trt.yml
│   │   └── config-llama2-70b-inf2-g5.yml
│   └── 7b
│       ├── config-llama2-7b-byo-sagemaker-endpoint.yml
│       ├── config-llama2-7b-g4dn-g5-trt.yml
│       ├── config-llama2-7b-g5-no-s3-quick.yml
│       ├── config-llama2-7b-g5-quick.yml
│       └── config-llama2-7b-inf2-g5.yml
├── llama3
│   ├── 70b
│   │   ├── config-bedrock.yml -> ../../bedrock/config-bedrock.yml
│   │   ├── config-ec2-llama3-70b-instruct.yml
│   │   ├── config-llama3-70b-instruct-g5-48xl.yml
│   │   ├── config-llama3-70b-instruct-g5-p4d.yml
│   │   └── config-llama3-70b-instruct-p4d.yml
│   └── 8b
│       ├── config-bedrock.yml
│       ├── config-ec2-llama3-8b-inf2-48xl.yml
│       ├── config-ec2-llama3-8b.yml
│       ├── config-llama3-8b-eks-inf2.yml
│       ├── config-llama3-8b-g5-streaming.yml
│       ├── config-llama3-8b-inf2-24xl-tp=8-bs=4-byoe.yml
│       ├── config-llama3-8b-inf2-48xl-tp=8-bs=4-byoe.yml
│       ├── config-llama3-8b-inf2-g5-byoe-w-openorca.yml
│       ├── config-llama3-8b-inf2-g5.yml
│       ├── config-llama3-8b-instruct-all.yml
│       ├── config-llama3-8b-instruct-g5-12xl-4-instances.yml
│       ├── config-llama3-8b-instruct-g5-12xl.yml
│       ├── config-llama3-8b-instruct-g5-24xl.yml
│       ├── config-llama3-8b-instruct-g5-2xl.yml
│       ├── config-llama3-8b-instruct-g5-48xl.yml
│       ├── config-llama3-8b-instruct-g5-p4d.yml
│       ├── config-llama3-8b-instruct-g6-12xl.yml
│       ├── config-llama3-8b-instruct-g6-24xl.yml
│       ├── config-llama3-8b-instruct-g6-48xl.yml
│       ├── config-llama3-8b-instruct-p4d-djl-lmi-dist.yml
│       ├── config-llama3-8b-instruct-p4d-djl-vllm.yml
│       ├── config-llama3-8b-instruct-p5-djl-lmi-dist.yml
│       ├── config-llama3-8b-trn1-32xl-tp=16-bs=4-byoe.yml
│       ├── config-llama3-8b-trn1-32xl-tp=8-bs=4-byoe.yml
│       ├── config-llama3-8b-trn1.yml
│       ├── llama3-8b-inf2-24xl-byoe-g5-12xl.yml
│       ├── llama3-8b-inf2-48xl-byoe-g5-24xl.yml
│       └── llama3-8b-trn1-32xl-byoe-g5-24xl.yml
├── mistral
│   ├── config-mistral-7b-eks-inf2.yml
│   ├── config-mistral-7b-tgi-g5.yml
│   ├── config-mistral-instruct-AWQ-p4d.yml
│   ├── config-mistral-instruct-AWQ-p5-byo-ep.yml
│   ├── config-mistral-instruct-AWQ-p5.yml
│   ├── config-mistral-instruct-p4d.yml
│   ├── config-mistral-instruct-v1-p5-trtllm.yml
│   ├── config-mistral-instruct-v2-p4d-lmi-dist.yml
│   ├── config-mistral-instruct-v2-p4d-trtllm.yml
│   ├── config-mistral-instruct-v2-p5-lmi-dist.yml
│   └── config-mistral-instruct-v2-p5-trtllm.yml
├── phi
│   └── config-phi-3-g5.yml
└── pricing.yml

```
