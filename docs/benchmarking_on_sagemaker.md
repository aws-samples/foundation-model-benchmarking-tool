# Benchmark models on SageMaker

Choose any config file from the model specific folders, for example the [`Llama3`](https://github.com/aws-samples/foundation-model-benchmarking-tool/tree/main/src/fmbench/configs/llama3) folder for `Llama3` family of models. These configuration files also include instructions for `FMBench` to first deploy the model on SageMaker using your configured instance type and inference parameters of choice and then run the benchmarking. Here is an example for benchmarking `Llama3-8b` model on an `ml.inf2.24xlarge` and `ml.g5.12xlarge` instance. 

```{.bash}
fmbench --config-file https://raw.githubusercontent.com/aws-samples/foundation-model-benchmarking-tool/main/src/fmbench/configs/llama3/8b/config-llama3-8b-inf2-g5.yml > fmbench.log 2>&1
```
