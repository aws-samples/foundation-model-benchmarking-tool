# Simplified and Parameterized Config Files for FMBench

## Introduction

Benchmarking multiple models across various configurations on Amazon EC2 used to require creating and managing multiple configuration files. Now, with parameterized config files, you can manage deployments, inference, and benchmarking with a **single configuration file** and simply change the parameters (such as the `instance type`, `tp degree`, `batch size`, `tokenizer directory`, `prompt template`, `model id`) all via the command line.

This approach eliminates redundancy and streamlines benchmarking processes for models deployed via **DJL on EC2**, with support for various instances, TP degrees, and batch sizes.

## Example: DJL Deployment Config File

Below is a generic config file for deploying models with DJL on EC2. Users can pass parameters dynamically to customize deployments.

```yaml
experiments:
  - name: {model_id}
    model_id: {model_id}
    ep_name: 'http://127.0.0.1:8080/invocations'
    instance_type: {instance_type}
    image_uri: 763104351884.dkr.ecr.{region}.amazonaws.com/djl-inference:0.29.0-lmi11.0.0-cu124
    deploy: yes
    instance_count: 
    deployment_script: ec2_deploy.py
    inference_script: ec2_predictor.py
    ec2:
      model_loading_timeout: 2400
    inference_spec:
      parameter_set: ec2_djl
      tp_degree: {tp_degree}
      shm_size: 12g
    serving.properties: |
      option.tensor_parallel_degree={tp_degree}
      option.max_rolling_batch_size={batch_size}
      option.model_id={model_id}
    payload_files:
    - payload_en_1-500.jsonl
    concurrency_levels:
    - 1
```

Now, you can deploy any models using the [standard djl configuration file](https://github.com/aws-samples/foundation-model-benchmarking-tool/blob/main/src/fmbench/configs/generic/ec2/djl.yml). 

## Command Syntax

Dynamic parameters can be passed during runtime. Here's an example command:

```{bash}
fmbench --config-file $CONFIG_FILE_PATH \
        --local-mode yes \
        --write-bucket placeholder \
        --tmp-dir /tmp \
        -A model_id=mistralai/Mistral-7B-Instruct-v0.2 \ # Mention your model id and other additional parameters below
        -A instance_type=g6e.4xlarge \ 
        -A tp_degree=1 \
        -A batch_size=4 \
        -A results_dir=Mistral-7B-Instruct-g6e.4xl \
        -A tokenizer_dir=mistral_tokenizer \
        -A prompt_template=prompt_template_mistral.txt \
        > $LOGFILE 2>&1
```

For other generic configuration files, view [here](https://github.com/aws-samples/foundation-model-benchmarking-tool/tree/main/src/fmbench/configs/generic).
