# Benchmark models on Bedrock

Choose any config file from the [`bedrock`](https://github.com/aws-samples/foundation-model-benchmarking-tool/tree/main/src/fmbench/configs/bedrock) folder and either run these directly or use them as templates for creating new config files specific to your use-case. Here is an example for benchmarking the `Llama3.1` models on Bedrock.

```{.bash}
fmbench --config-file https://raw.githubusercontent.com/aws-samples/foundation-model-benchmarking-tool/main/src/fmbench/configs/bedrock/config-bedrock-llama3-1.yml > fmbench.log 2>&1
```
