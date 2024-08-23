## 2.0.0

1. 🚨 Model evaluations done by a **Panel of LLM Evaluators[[1]](#1)** 🚨

## v1.0.52

1. Compile for AWS Chips (Trainium, Inferentia) and deploy to SageMaker directly through `FMBench`.
1. `Llama3.1-8b` and `Llama3.1-70b` config files for AWS Chips (Trainium, Inferentia).
1. Misc. bug fixes.

## v1.0.51

1. `FMBench` has a [website](https://aws-samples.github.io/foundation-model-benchmarking-tool/index.html) now. Rework the README file to make it lightweight.
1. `Llama3.1` config files for Bedrock.

### v1.0.50
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

### v1.0.47
1. Run `FMBench` as a Docker container.
1. Bug fixes for GovCloud support.
1. Updated README for EKS cluster creation.

### v1.0.46
1. Native model deployment support for EC2 and EKS (i.e. you can now deploy and benchmark models on EC2 and EKS).
1. FMBench is now available in GovCloud.
1. Update to latest version of several packages.

### v1.0.45
1. Analytics for results across multiple runs.
1. `Llama3-70b` config files for `g5.48xlarge` instances.

### v1.0.44
1. Endpoint metrics (CPU/GPU utilization, memory utiliztion, model latency) and invocation metrics (including errors) for SageMaker Endpoints.
1. `Llama3-8b` config files for `g6` instances.

### v1.0.42
1. Config file for running `Llama3-8b` on all instance types except `p5`.
1. Fix bug with business summary chart.
1. Fix bug with deploying model using a DJL DeepSpeed container in the no S3 dependency mode.

### v1.0.40
1. Make it easy to run in the Amazon EC2 without any dependency on Amazon S3 dependency mode.

### v1.0.39
1. Add an internal `FMBench` website.

### v1.0.38
1. Support for running `FMBench` on Amazon EC2 without any dependency on Amazon S3.
1. [`Llama3-8b-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) config file for `ml.p5.48xlarge`.

### v1.0.37
1. `g5`/`p4d`/`inf2`/`trn1` specific config files for [`Llama3-8b-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).
    1. `p4d` config file for both `vllm` and `lmi-dist`.

### v1.0.36
1. Fix bug at higher concurrency levels (20 and above).
1. Support for instance count > 1.


### v1.0.35

1. Support for [Open-Orca](https://huggingface.co/datasets/Open-Orca/OpenOrca) dataset and corresponding prompts for Llama3, Llama2 and Mistral.

### v1.0.34
1. Don't delete endpoints for the bring your own endpoint case.
1. Fix bug with business summary chart.

### v1.0.32

1. Report enhancements: New business summary chart, config file embedded in the report, version numbering and others.

1. Additional config files: Meta Llama3 on Inf2, Mistral instruct with `lmi-dist` on `p4d` and `p5` instances.
