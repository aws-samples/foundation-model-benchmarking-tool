### Benchmarking Multimodal Models on Amazon Bedrock

FMBench now enables customers to benchmark multimodal models available through Amazon Bedrock. This feature supports both Claude and Llama 3.2 models, allowing you to evaluate their performance on multimodal tasks. Currently, FMBench supports multimodal benchmarking for: `Anthropic Claude` and `Meta Llama 3.2` Vision models.

#### Prerequisites

Before running multimodal benchmarks, ensure you have:

1. Enabled model access to Llama 3.2 models in your Amazon Bedrock console.

#### Running Multimodal Benchmarks on FMBench

To benchmark multimodal models on Amazon Bedrock, use the provided configuration files. Here's an example command:

``` {.bash}
fmbench --config-file https://raw.githubusercontent.com/aws-samples/foundation-model-benchmarking-tool/main/src/fmbench/configs/bedrock/config-llama-3-2-11b-vision-instruct-scienceqa.yml > fmbench.log 2>&1
```

##### **This command will**:

1. Load the specified configuration file for Llama 3.2 11B Vision model

1. Run the benchmark using the `derek-thomas/ScienceQA` dataset.