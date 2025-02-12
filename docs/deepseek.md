# DeepSeek-R1

The distilled version of Deepseek-R1 models are now supported for both performance benchmarking and model evaluations 🎉. You can use built in support for 4 different datasets: [`LongBench`](https://huggingface.co/datasets/THUDM/LongBench), [`Dolly`](https://huggingface.co/datasets/databricks/databricks-dolly-15k), [`OpenOrca`](https://huggingface.co/datasets/Open-Orca/OpenOrca) and [`ConvFinQA`](https://huggingface.co/datasets/AdaptLLM/finance-tasks/tree/refs%2Fconvert%2Fparquet/ConvFinQA). You can deploy the Deepseek-R1 distilled models on Amazon EC2, Amazon Bedrock or Amazon SageMaker.

The easiest way to benchmark the DeepSeek models is through the [`FMBench-orchestrator`](https://github.com/awslabs/fmbench-orchestrator) on Amazon EC2 VMs.

## Benchmark Deepseek-R1 distilled models on Amazon EC2

👉 Make sure your account has enough service quota for vCPUs to run this benchmark. We would be using `g6e.xlarge`, `g6e.2xlarge`, `g6e.12xlarge` and `g6e.48xlarge` instances, if you do not have sufficient service quota then you can set `deploy: no` in the `configs/deepseek/deepseek-convfinqa.yml` (or other) file to disable some tests as needed.

Follow instructions [here](https://github.com/awslabs/fmbench-orchestrator?tab=readme-ov-file#install-fmbench-orchestrator-on-ec2) to install the orchestrator. Once installed you can run Deepseek-r1 benchmarking with the [`ConvFinQA`](https://huggingface.co/datasets/AdaptLLM/finance-tasks/tree/refs%2Fconvert%2Fparquet/ConvFinQA) dataset the following command:

```{.bashrc}
python main.py --config-file configs/deepseek/deepseek-convfinqa.yml
```
Change the `--config-file` parameter to [`configs/deepseek/deepseek-longbench.yml`](https://github.com/aws-samples/fmbench-orchestrator/blob/main/configs/deepseek/deepseek-longbench.yml) or [`configs/deepseek/deepseek-openorca.yml`](https://github.com/aws-samples/fmbench-orchestrator/blob/main/configs/deepseek/deepseek-openorca.yml) to use other datasets for benchmarking. These orchestrator files test various Deepseek-R1 distilled models on `g6e` instances, edit this file as per your requirements. 

## Benchmark Deepseek-R1 quantized models on Amazon EC2

👉 Make sure your account has enough service quota for vCPUs to run this benchmark. We would be using `g6e.12xlarge` instance for this test.


1. Create a `g6e.12xlarge` instance and run the `DeepSeek-R1 1.58b quantized` model on this instance by following the steps 1 through 8 described [here](https://github.com/aarora79/deepseek-r1-ec2?tab=readme-ov-file#quantized-models).

1. Follow steps 1 through 5 [here](https://aws-samples.github.io/foundation-model-benchmarking-tool/benchmarking_on_ec2.html#benchmarking-on-an-instance-type-with-nvidia-gpus-or-aws-chips) to setup `FMBench` on this instance.

1. Next run the following command to benchmark LongBench 

    ```{.bashrc}
    TMP_DIR=/tmp
    fmbench --config-file $TMP_DIR/fmbench-read/configs/deepseek/config-deepseek-r1-quant1.58-longbench-byoe.yml --local-mode yes --write-bucket placeholder --tmp-dir $TMP_DIR > fmbench.log 2>&1
    ```

1. Once the run completes you should see the benchmarking results in a folder called `results-DeepSeek-R1-quant-1.58bit-g6e.12xl` present in your current directory.
