
# Results for performance benchmarking llama3-8b-instruct

|**Last modified (UTC)** | **FMBench version**  |
|---|---|
|2025-01-26 16:47:29.483491|2.0.27|


## Summary

We did performance benchmarking for `llama3-8b-instruct` on "`g5.2xlarge`" on multiple datasets and based on the test results the best price performance for dataset `en_3000-3840` is provided by the `g5.2xlarge`.


| Information | Value |
|-----|-----|
| experiment_name | llama3-8b-instruct |
| payload_file | payload_en_3000-3840.jsonl |
| instance_type | g5.2xlarge |
| instance_count | 1.0 |
| concurrency | 1 |
| error_rate | 0.0 |
| prompt_token_count_mean | 3424 |
| prompt_token_throughput | 5935 |
| completion_token_count_mean | 15 |
| completion_token_throughput | 20 |
| transactions_per_minute | 103 |
| price_per_txn | 0.000196 |
| price_per_token | 0.00000006 |
| latency p50, p95, p99 | 0.64, 0.64, 0.64 |



The price performance comparison for different instance types is presented below. An interactive version of this chart is available [here](llama3-8b-g5.html).

![Price performance comparison](business_summary.png)

### Latency Metrics Analysis

The following table provides token latency metrics including the overall latency, Time To First Token (TTFT), and Time Per Output Token (TPOT) for the `en_3000-3840` dataset.


_No token latency metrics data are available. To get token latency metrics, ensure your inference endpoint allows streaming responses, and enable streaming responses by setting `stream: True` in experiment configuration_.


### Failed experiments


All experiments satisfied the configured performance criteria: `Latency` < `2s`, `cost per 10k transactions`: `$20`, `error rate`: `0`.






### Model evaluations

_Model evaluation data is not available_.

### Endpoint metrics

The following table provides endpoint utilization and invocation metrics.

| percentile | CPU Utilization % | Memory Utilization % | Memory Used (GB) | GPU Utilization % | GPU Memory Used (GB) | GPU Memory Free (GB) | GPU Memory Total (GB) |
|-----|-----|-----|-----|-----|-----|-----|-----|
| 0% | 0.0 | 27.7 | 3.73 | 0.0 | 40312.38 | 4132.36 | 46068.0 |
| 25% | 26.62 | 27.73 | 3.73 | 88.87 | 40484.13 | 4508.73 | 46068.0 |
| 50% | 26.95 | 27.75 | 3.74 | 90.31 | 40686.63 | 4686.05 | 46068.0 |
| 75% | 27.13 | 27.76 | 3.74 | 90.53 | 40863.96 | 4888.56 | 46068.0 |
| 100% | 27.37 | 27.77 | 3.74 | 91.51 | 41240.32 | 5060.31 | 46068.0 |


### Configuration

The configuration used for these tests is available in the [`config`](#configuration-file) file.

### Experiment cost

#### Model Benchmarking Cost

The cost to run each experiment is provided in the table below. The total cost for running all experiments is $0.03.


| experiment_name | instance_type | instance_count | duration_in_seconds | cost |
|-----|-----|-----|-----|-----|
| llama3-8b-instruct | g5.2xlarge | nan | 96.56 | 0.03251 |


#### Model Evaluation Cost

The cost to evaluate 1 candidate models using No LLM evaluators is 0.



The total cost incurred for **model benchmarking and evaluations**: $0.03.


## Per instance results

The following table provides the best combinations for running inference for different sizes prompts on different instance types. The following dataset(s) were used for this test: `2wikimqa_e.jsonl`, `2wikimqa.jsonl`, `hotpotqa_e.jsonl`, `hotpotqa.jsonl`, `narrativeqa.jsonl`, `triviaqa_e.jsonl`, `triviaqa.jsonl`.

|Dataset   | Instance type   | Recommendation   |
|---|---|---|
|`payload_en_1-500.jsonl`|`g5.2xlarge`|The best option for staying within a latency budget of `2 seconds` on a `g5.2xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `median latency of 0.77 seconds`, for an `average prompt size of 284 tokens` and `completion size of 31 tokens` with `77 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`g5.2xlarge`|The best option for staying within a latency budget of `2 seconds` on a `g5.2xlarge` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `median latency of 0.59 seconds`, for an `average prompt size of 1566 tokens` and `completion size of 19 tokens` with `130 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`g5.2xlarge`|The best option for staying within a latency budget of `2 seconds` on a `g5.2xlarge` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `median latency of 0.61 seconds`, for an `average prompt size of 2536 tokens` and `completion size of 17 tokens` with `121 transactions/minute`.|
|`payload_en_3000-3840.jsonl`|`g5.2xlarge`|The best option for staying within a latency budget of `2 seconds` on a `g5.2xlarge` for the `payload_en_3000-3840.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `median latency of 0.64 seconds`, for an `average prompt size of 3424 tokens` and `completion size of 15 tokens` with `103 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`g5.2xlarge`|The best option for staying within a latency budget of `2 seconds` on a `g5.2xlarge` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `median latency of 0.64 seconds`, for an `average prompt size of 837 tokens` and `completion size of 24 tokens` with `101 transactions/minute`.|

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
  sagemaker_execution_role: arn:aws:iam::015469603702:role/fmbench-orchestrator
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
    max_length_in_tokens: 3840
    min_length_in_tokens: 3000
    payload_file: payload_en_3000-3840.jsonl
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
  instance_type: g5.2xlarge
  model_id: meta-llama/Meta-Llama-3-8B-Instruct
  model_name: llama3-8b-instruct
  model_version: null
  name: llama3-8b-instruct
  payload_files:
  - payload_en_1-500.jsonl
  - payload_en_500-1000.jsonl
  - payload_en_1000-2000.jsonl
  - payload_en_2000-3000.jsonl
  - payload_en_3000-3840.jsonl
  region: us-east-1
  serving.properties: 'engine=MPI

    option.tensor_parallel_degree=1

    option.max_rolling_batch_size=256

    option.model_id=meta-llama/Meta-Llama-3-8B-Instruct

    option.rolling_batch=lmi-dist

    '
general:
  model_name: llama3-8b-instruct
  name: llama3-8b-g5.2xl-djl-ec2
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
  dataset_of_interest: en_3000-3840
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
  1_generate_data.ipynb: true
  2_deploy_model.ipynb: true
  3_run_inference.ipynb: true
  4_model_metric_analysis.ipynb: true
  5_cleanup.ipynb: true
s3_read_data:
  config_files:
  - pricing.yml
  configs_prefix: configs
  local_file_system_path: /tmp/fmbench-read
  prompt_template_dir: prompt_template
  prompt_template_file: prompt_template_llama3.txt
  read_bucket: sagemaker-fmbench-read-us-east-1-015469603702
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
  tokenizer_prefix: llama3_tokenizer

```
