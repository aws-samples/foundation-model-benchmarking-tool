# DeepSeek-R1

The distilled version of Deepseek-R1 models are now supported for both performance benchmarking and model evaluations 🎉. You can use built in support for 4 different datasets: [`LongBench`](https://huggingface.co/datasets/THUDM/LongBench), [`Dolly`](https://huggingface.co/datasets/databricks/databricks-dolly-15k), [`OpenOrca`](https://huggingface.co/datasets/Open-Orca/OpenOrca) and [`ConvFinQA`](https://huggingface.co/datasets/AdaptLLM/finance-tasks/tree/refs%2Fconvert%2Fparquet/ConvFinQA). You can deploy the Deepseek-R1 distilled models on Amazon EC2, Amazon Bedrock or Amazon SageMaker.

The easiest way to benchmark the DeepSeek models is through the [`FMBench-orchestrator`](https://github.com/awslabs/fmbench-orchestrator) on Amazon EC2 VMs.

## Benchmark Deepseek-R1 distilled models on Amazon EC2

Follow instructions [here](https://github.com/awslabs/fmbench-orchestrator?tab=readme-ov-file#install-fmbench-orchestrator-on-ec2) to install the orchestrator. Once installed you can run Deepseek-r1 benchmarking with the [`ConvFinQA`](https://huggingface.co/datasets/AdaptLLM/finance-tasks/tree/refs%2Fconvert%2Fparquet/ConvFinQA) dataset the following command:

```{.bashrc}
python main.py --config-file configs/deepseek/deepseek-convfinqa.yml
```
Change the `--config-file` parameter to [`configs/deepseek/deepseek-longbench.yml`](https://github.com/aarora79/fmbench-orchestrator/blob/main/configs/deepseek/deepseek-longbench.yml) or [`configs/deepseek/deepseek-openorca.yml`](https://github.com/aarora79/fmbench-orchestrator/blob/main/configs/deepseek/deepseek-openorca.yml) to use other datasets for benchmarking. These orchestrator files test various Deepseek-R1 distilled models on `g6e` instances, edit this file as per your requirements.

