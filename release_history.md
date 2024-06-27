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
