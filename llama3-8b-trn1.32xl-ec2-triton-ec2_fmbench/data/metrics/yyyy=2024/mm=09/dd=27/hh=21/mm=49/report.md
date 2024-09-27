
# Results for performance benchmarking Meta-Llama-3-8B-Instruct

|**Last modified (UTC)** | **FMBench version**  |
|---|---|
|2024-09-27 21:57:47.663450|2.0.8|


## Summary

We did performance benchmarking for `Meta-Llama-3-8B-Instruct` on "`trn1.32xlarge`" on multiple datasets and based on the test results the best price performance for dataset `en_3000-3840` is provided by the `trn1.32xlarge`.


| Information | Value |
|-----|-----|
| experiment_name | Meta-Llama-3-8B-Instruct-triton |
| payload_file | payload_en_3000-3840.jsonl |
| instance_type | trn1.32xlarge |
| instance_count | 1.0 |
| concurrency | 14 |
| error_rate | 0.0 |
| prompt_token_count_mean | 3433 |
| prompt_token_throughput | 23420 |
| completion_token_count_mean | 41 |
| completion_token_throughput | 277 |
| transactions_per_minute | 409 |
| price_per_txn | 0.000876 |
| price_per_token | 0.00000025 |
| latency p50, p95, p99 | 1.38, 1.86, 2.05 |



The price performance comparison for different instance types is presented below. An interactive version of this chart is available [here](llama3-8b-trn1.html).

![Price performance comparison](business_summary.png)

### Latency Metrics Analysis

The following table provides token latency metrics including the overall latency, Time To First Token (TTFT), and Time Per Output Token (TPOT) for the `en_3000-3840` dataset.


_No token latency metrics data is available. To get token latency metrics enable streaming responses by setting `stream: True` in experiment configuration_.


### Failed experiments


All experiments satisfied the configured performance criteria: `Latency` < `2s`, `cost per 10k transactions`: `$100`, `error rate`: `0`.






### Model evaluations

_Model evaluation data is not available_.

### Endpoint metrics

The following table provides endpoint utilization and invocation metrics.

_No endpoint metrics data is available_.

### Configuration

The configuration used for these tests is available in the [`config`](#configuration-file) file.

### Experiment cost

#### Model Benchmarking Cost

The cost to run each experiment is provided in the table below. The total cost for running all experiments is $2.57.


| experiment_name | instance_type | instance_count | duration_in_seconds | cost |
|-----|-----|-----|-----|-----|
| Meta-Llama-3-8B-Instruct-triton | trn1.32xlarge | nan | 430.5 | 2.571061 |


#### Model Evaluation Cost

The cost to evaluate 1 candidate models using No LLM evaluators is 0.



The total cost incurred for **model benchmarking and evaluations**: $2.57.


## Per instance results

The following table provides the best combinations for running inference for different sizes prompts on different instance types. The following dataset(s) were used for this test: `2wikimqa_e.jsonl`, `2wikimqa.jsonl`, `hotpotqa_e.jsonl`, `hotpotqa.jsonl`, `narrativeqa.jsonl`, `triviaqa_e.jsonl`, `triviaqa.jsonl`.

|Dataset   | Instance type   | Recommendation   |
|---|---|---|
|`payload_en_1-500.jsonl`|`trn1.32xlarge`|The best option for staying within a latency budget of `2 seconds` on a `trn1.32xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 14`. A concurrency level of 14 achieves an `median latency of 1.33 seconds`, for an `average prompt size of 274 tokens` and `completion size of 40 tokens` with `603 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`trn1.32xlarge`|The best option for staying within a latency budget of `2 seconds` on a `trn1.32xlarge` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 14`. A concurrency level of 14 achieves an `median latency of 1.35 seconds`, for an `average prompt size of 1552 tokens` and `completion size of 48 tokens` with `423 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`trn1.32xlarge`|The best option for staying within a latency budget of `2 seconds` on a `trn1.32xlarge` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 14`. A concurrency level of 14 achieves an `median latency of 1.42 seconds`, for an `average prompt size of 2520 tokens` and `completion size of 51 tokens` with `386 transactions/minute`.|
|`payload_en_3000-3840.jsonl`|`trn1.32xlarge`|The best option for staying within a latency budget of `2 seconds` on a `trn1.32xlarge` for the `payload_en_3000-3840.jsonl` dataset is a `concurrency level of 14`. A concurrency level of 14 achieves an `median latency of 1.38 seconds`, for an `average prompt size of 3433 tokens` and `completion size of 41 tokens` with `409 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`trn1.32xlarge`|The best option for staying within a latency budget of `2 seconds` on a `trn1.32xlarge` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 14`. A concurrency level of 14 achieves an `median latency of 1.47 seconds`, for an `average prompt size of 827 tokens` and `completion size of 47 tokens` with `551 transactions/minute`.|

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
  region: us-west-2
  s3_and_or_local_file_system: local
  sagemaker_execution_role: arn:aws:iam::218208277580:role/ec2_fmbench
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
  - 4
  - 8
  - 10
  - 12
  - 14
  deploy: true
  deployment_script: ec2_deploy.py
  ec2:
    gpu_or_neuron_setting: -u triton --device /dev/neuron0 --device /dev/neuron1 --device
      /dev/neuron2 --device /dev/neuron3 --device /dev/neuron4 --device /dev/neuron5
      --device /dev/neuron6 --device /dev/neuron7 --device /dev/neuron8 --device /dev/neuron9
      --device /dev/neuron10 --device /dev/neuron11 --device /dev/neuron12 --device
      /dev/neuron13 --device /dev/neuron14 --device /dev/neuron15
    model_loading_timeout: 8000
  env: null
  ep_name: http://localhost:8080/v2/models/Meta-Llama-3-8B-Instruct/generate
  image_uri: tritonserver-neuronx:fmbench
  inference_script: ec2_predictor.py
  inference_spec:
    backend: djl
    container_type: triton
    max_model_len: 4096
    model_copies: max
    model_loading_timeout: 2400
    model_params:
      amp: f16
      attention_layout: BSH
      collectives_layout: BSH
      context_length_estimate: 3072, 3584, 4096
      max_rolling_batch_size: 8
      model_loader: tnx
      model_loading_timeout: 2400
      n_positions: 4096
      output_formatter: json
      rolling_batch: auto
      rolling_batch_strategy: continuous_batching
      trust_remote_code: true
    parameter_set: ec2_djl
    parameters:
      max_tokens: 100
      top_k: 50
    shm_size: 12g
    tp_degree: 8
  instance_count: null
  instance_type: trn1.32xlarge
  model_id: meta-llama/Meta-Llama-3-8B-Instruct
  model_name: Meta-Llama-3-8B-Instruct
  model_version: null
  name: Meta-Llama-3-8B-Instruct-triton
  payload_files:
  - payload_en_1-500.jsonl
  - payload_en_500-1000.jsonl
  - payload_en_1000-2000.jsonl
  - payload_en_2000-3000.jsonl
  - payload_en_3000-3840.jsonl
  region: us-west-2
  serving.properties: null
general:
  model_name: Meta-Llama-3-8B-Instruct
  name: llama3-8b-trn1.32xl-ec2-triton
inference_parameters:
  ec2_djl:
    max_tokens: 100
    top_k: 50
metrics:
  dataset_of_interest: en_3000-3840
pricing: pricing.yml
report:
  all_metrics_file: all_metrics.csv
  cost_per_10k_txn_budget: 100
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
  read_bucket: sagemaker-fmbench-read-us-west-2-218208277580
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
