## Models benchmarked

Configuration files are available in the [configs](./src/fmbench/configs) folder for the following models in this repo.

### Llama3 on Amazon SageMaker

Llama3 is now available on SageMaker (read [blog post](https://aws.amazon.com/blogs/machine-learning/meta-llama-3-models-are-now-available-in-amazon-sagemaker-jumpstart/)), and you can now benchmark it using `FMBench`. Here are the config files for benchmarking `Llama3-8b-instruct` and `Llama3-70b-instruct` on `ml.p4d.24xlarge`, `ml.inf2.24xlarge` and `ml.g5.12xlarge` instances.

- [Config file](https://github.com/aws-samples/foundation-model-benchmarking-tool/blob/main/src/fmbench/configs/llama3/8b/config-llama3-8b-instruct-g5-p4d.yml) for `Llama3-8b-instruct` on  `ml.p4d.24xlarge` and `ml.g5.12xlarge`.
- [Config file](https://github.com/aws-samples/foundation-model-benchmarking-tool/blob/main/src/fmbench/configs/llama3/70b/config-llama3-70b-instruct-g5-p4d.yml) for `Llama3-70b-instruct` on  `ml.p4d.24xlarge` and `ml.g5.48xlarge`.
- [Config file](https://github.com/aws-samples/foundation-model-benchmarking-tool/blob/main/src/fmbench/configs/llama3/8b/config-llama3-8b-inf2-g5.yml) for `Llama3-8b-instruct` on  `ml.inf2.24xlarge` and `ml.g5.12xlarge`.

### Full list of benchmarked models

| Model    | SageMaker g4dn/g5/p3 | SageMaker Inf2 | SageMaker P4 | SageMaker P5 | Bedrock On-demand throughput | Bedrock provisioned throughput |
|:------------------|:-----------------|:----------------|:--------------|:--------------|:------------------------------|:--------------------------------|
| **Anthropic Claude-3 Sonnet** | | |  | | ✅ | ✅  | 
| **Anthropic Claude-3 Haiku**  | | |  | | ✅ |   |
| **Mistral-7b-instruct** |✅ | |✅  |✅ | ✅ |   |
| **Mistral-7b-AWQ** || | |✅ | |   |
| **Mixtral-8x7b-instruct**  | | |  | | ✅ |   |
| **Llama3-8b instruct**  |✅ |✅|✅  | ✅|✅  |   |
| **Llama3-70b instruct**  |✅ |✅|✅  | |✅ |   |
| **Llama2-13b chat**  |✅ |✅ |✅  | | ✅  |   |
| **Llama2-70b chat**  |✅ |✅ |✅  | | ✅  |   |
| **Amazon Titan text lite**  | | |  | | ✅ |   |
| **Amazon Titan text express**  | | |  | | ✅ |   |
| **Cohere Command text**  | | |  | | ✅ |   |
| **Cohere Command light text**  | | |  | | ✅ |   |
| **AI21 J2 Mid**  | | |  | | ✅ |   |
| **AI21 J2 Ultra** | | |  | | ✅ |   |
| **Gemma-2b** |✅ | |  | |  |   |
| **Phi-3-mini-4k-instruct** |✅ | |  | |  |   |
| **distilbert-base-uncased**  |  ✅ | |  | ||   |

## New in this release

## v1.0.50
1. `Llama3-8b` on Amazon EC2 `inf2.48xlarge` config file.
1. Update to new version of DJL LMI (0.28.0).

### v1.0.49
1. Streaming support for Amazon SageMaker and Amazon Bedrock.
1. Per-token latency metrics such as time to first token (TTFT) and mean time per-output token (TPOT).
1. Misc. bug fixes.

### v1.0.48
1. Faster result file download at the end of a test run.
1. `Phi-3-mini-4k-instruct` configuration file.
1. Tokenizer and misc. bug fixes.


[Release history](./release_history.md)