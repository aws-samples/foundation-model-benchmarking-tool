
# Results for performance benchmarking llama3-1-8b-instruct

|**Last modified (UTC)** | **FMBench version**  |
|---|---|
|2024-11-29 15:57:16.656251|2.0.20|


## Summary

We did performance benchmarking for `llama3-1-8b-instruct` on "`g5.4xlarge`" on multiple datasets and based on the test results the best price performance for dataset `en_500-1000` is provided by the `g5.4xlarge`.


| Information | Value |
|-----|-----|
| experiment_name | llama3-1-8b-instruct |
| payload_file | payload_en_500-1000.jsonl |
| instance_type | g5.4xlarge |
| instance_count | 1.0 |
| concurrency | 1 |
| error_rate | 0.0 |
| prompt_token_count_mean | 837 |
| prompt_token_throughput | 504 |
| completion_token_count_mean | 40 |
| completion_token_throughput | 23 |
| transactions_per_minute | 35 |
| price_per_txn | 0.000773 |
| price_per_token | 0.00000088 |
| latency p50, p95, p99 | 1.68, 1.68, 1.68 |



The price performance comparison for different instance types is presented below. An interactive version of this chart is available [here](llama3-1-8b-g5.html).

![Price performance comparison](business_summary.png)

### Latency Metrics Analysis

The following table provides token latency metrics including the overall latency, Time To First Token (TTFT), and Time Per Output Token (TPOT) for the `en_500-1000` dataset.


_No token latency metrics data is available. To get token latency metrics enable streaming responses by setting `stream: True` in experiment configuration_.


### Failed experiments


There were a total of 1 experiment run(s) that failed at least one configured performance criteria: `Latency` < `2s`, `cost per 10k transactions`: `$20`, `error rate`: `0`. See table below.    
    



| experiment_name | payload_file | concurrency | error_rate_text | latency_p95_text | price_per_10k_txn_text |
|-----|-----|-----|-----|-----|-----|
| llama3-1-8b-instruct | payload_en_500-1000.jsonl | 2 | <span style='color:green'>0.00</span> | <span style='color:red'>**2.14**</span> | <span style='color:green'>4.92</span> |



### Model evaluations

_Model evaluation data is not available_.

### Endpoint metrics

The following table provides endpoint utilization and invocation metrics.

| Unnamed: 0 | min | 25% | 50% | 75% | max |
|-----|-----|-----|-----|-----|-----|
| cpu_percent_mean | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| memory_percent_mean | 13.0 | 13.0 | 13.0 | 13.0 | 13.0 |
| memory_used_mean | 3.35 | 3.35 | 3.35 | 3.35 | 3.35 |
| gpu_utilization_mean | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gpu_memory_used_mean | 19678.0 | 19678.0 | 19678.0 | 19678.0 | 19678.0 |
| gpu_memory_free_mean | 3038.5 | 3038.5 | 3038.5 | 3038.5 | 3038.5 |
| gpu_memory_total_mean | 23028.0 | 23028.0 | 23028.0 | 23028.0 | 23028.0 |


### Configuration

The configuration used for these tests is available in the [`config`](#configuration-file) file.

### Experiment cost

#### Model Benchmarking Cost

The cost to run each experiment is provided in the table below. The total cost for running all experiments is $0.01.


| experiment_name | instance_type | instance_count | duration_in_seconds | cost |
|-----|-----|-----|-----|-----|
| llama3-1-8b-instruct | g5.4xlarge | nan | 30.59 | 0.013801 |


#### Model Evaluation Cost

The cost to evaluate 1 candidate models using No LLM evaluators is 0.



The total cost incurred for **model benchmarking and evaluations**: $0.01.


## Per instance results

The following table provides the best combinations for running inference for different sizes prompts on different instance types. The following dataset(s) were used for this test: `2wikimqa_e.jsonl`, `2wikimqa.jsonl`, `hotpotqa_e.jsonl`, `hotpotqa.jsonl`, `narrativeqa.jsonl`, `triviaqa_e.jsonl`, `triviaqa.jsonl`.

|Dataset   | Instance type   | Recommendation   |
|---|---|---|
|`payload_en_1-500.jsonl`|`g5.4xlarge`|The best option for staying within a latency budget of `2 seconds` on a `g5.4xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `median latency of 1.18 seconds`, for an `average prompt size of 284 tokens` and `completion size of 25 tokens` with `101 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`g5.4xlarge`|The best option for staying within a latency budget of `2 seconds` on a `g5.4xlarge` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `median latency of 1.68 seconds`, for an `average prompt size of 837 tokens` and `completion size of 40 tokens` with `35 transactions/minute`.|

## Plots

The following plots provide insights into the results from the different experiments run.

![Error rates for different concurrency levels and instance types](error_rates.png)

![Tokens vs latency for different concurrency levels and instance types](tokens_vs_latency.png)

![Concurrency Vs latency for different instance type for selected dataset](concurrency_vs_inference_latency.png)



## Configuration file
```.bash
aws:
  bucket: placeholder
  local_file_system_path: /tmp/fmbench-write
  region: us-east-1
  s3_and_or_local_file_system: local
  sagemaker_execution_role: arn:aws:iam::988564344122:role/ec2adminrole
datasets:
  filters:
  - language: en
    max_length_in_tokens: 500
    min_length_in_tokens: 1
    payload_file: payload_en_1-500.jsonl
  - language: en
    max_length_in_tokens: 1000
    min_length_in_tokens: 500
    payload_file: payload_en_500-1000.jsonl
  - language: en
    max_length_in_tokens: 2000
    min_length_in_tokens: 1000
    payload_file: payload_en_1000-2000.jsonl
  - language: en
    max_length_in_tokens: 3000
    min_length_in_tokens: 2000
    payload_file: payload_en_2000-3000.jsonl
  - language: en
    max_length_in_tokens: 4000
    min_length_in_tokens: 3000
    payload_file: payload_en_3000-4000.jsonl
  - language: en
    max_length_in_tokens: 3840
    min_length_in_tokens: 3000
    payload_file: payload_en_3000-3840.jsonl
  - language: en
    max_length_in_tokens: 924
    min_length_in_tokens: 1
    payload_file: payload_en_1-924.jsonl
  prompt_template_keys:
  - input
  - context
dir_paths:
  all_prompts_file: all_prompts.csv
  data_prefix: data
  metadata_dir: metadata
  metrics_dir: metrics
  models_dir: models
  prompts_prefix: prompts
experiments:
- bucket: placeholder
  concurrency_levels:
  - 1
  - 2
  deploy: true
  deployment_script: ec2_deploy.py
  ec2:
    gpu_or_neuron_setting: --gpus all --shm-size 12g
    model_loading_timeout: 2400
  env: null
  ep_name: http://127.0.0.1:8080/invocations
  image_uri: 763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.29.0-lmi11.0.0-cu124
  inference_script: ec2_predictor.py
  inference_spec:
    model_copies: max
    model_loading_timeout: 2400
    parameter_set: ec2_djl
    parameters:
      Content-type: application/json
      do_sample: true
      max_new_tokens: 100
      return_full_text: false
      temperature: 0.1
      top_k: 120
      top_p: 0.92
    shm_size: 12g
    tp_degree: 1
  instance_count: null
  instance_type: g5.4xlarge
  model_id: meta-llama/Llama-3.1-8B-Instruct
  model_name: llama3-1-8b-instruct
  model_version: null
  name: llama3-1-8b-instruct
  payload_files:
  - payload_en_1-500.jsonl
  - payload_en_500-1000.jsonl
  region: us-east-1
  serving.properties: 'engine=MPI

    option.tensor_parallel_degree=1

    option.model_id=meta-llama/Meta-Llama-3.1-8b-Instruct

    option.rolling_batch=lmi-dist

    '
general:
  model_name: llama3-1-8b-instruct
  name: llama3-1-8b-g5.4xl-ec2
inference_parameters:
  ec2_djl:
    Content-type: application/json
    do_sample: true
    max_new_tokens: 100
    return_full_text: false
    temperature: 0.1
    top_k: 120
    top_p: 0.92
metrics:
  dataset_of_interest: en_500-1000
pricing: pricing.yml
report:
  all_metrics_file: all_metrics.csv
  cost_per_10k_txn_budget: 20
  error_rate_budget: 0
  latency_budget: 2
  per_inference_request_file: per_inference_request_results.csv
  txn_count_for_showing_cost: 10000
  v_shift_w_gt_one_instance: 0.025
  v_shift_w_single_instance: 0.025
run_steps:
  0_setup.ipynb: true
  1_generate_data.ipynb: false
  2_deploy_model.ipynb: true
  3_run_inference.ipynb: true
  4_model_metric_analysis.ipynb: true
  5_cleanup.ipynb: false
s3_read_data:
  config_files:
  - pricing.yml
  configs_prefix: configs
  local_file_system_path: /tmp/fmbench-read
  prompt_template_dir: prompt_template
  prompt_template_file: prompt_template_llama3.txt
  read_bucket: sagemaker-fmbench-read-us-east-1-988564344122
  s3_or_local_file_system: local
  script_files:
  - hf_token.txt
  scripts_prefix: scripts
  source_data_files:
  - 2wikimqa_e.jsonl
  - 2wikimqa.jsonl
  - hotpotqa_e.jsonl
  - hotpotqa.jsonl
  - narrativeqa.jsonl
  - triviaqa_e.jsonl
  - triviaqa.jsonl
  source_data_prefix: source_data
  tokenizer_prefix: llama3_1_tokenizer

```
