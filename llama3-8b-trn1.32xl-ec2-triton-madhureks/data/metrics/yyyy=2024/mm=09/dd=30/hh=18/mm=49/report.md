
# Results for performance benchmarking Meta-Llama-3-8B-Instruct

|**Last modified (UTC)** | **FMBench version**  |
|---|---|
|2024-09-30 19:12:03.438135|2.0.8|


## Summary

We did performance benchmarking for `Meta-Llama-3-8B-Instruct` on "`trn1.32xlarge`" on multiple datasets and based on the test results the best price performance for dataset `en_3000-3840` is provided by the `trn1.32xlarge`.


| Information | Value |
|-----|-----|
| experiment_name | Meta-Llama-3-8B-Instruct-triton |
| payload_file | payload_en_3000-3840.jsonl |
| instance_type | trn1.32xlarge |
| instance_count | 1.0 |
| concurrency | 2 |
| error_rate | 0.0 |
| prompt_token_count_mean | 3433 |
| prompt_token_throughput | 4596 |
| completion_token_count_mean | 42 |
| completion_token_throughput | 46 |
| transactions_per_minute | 79 |
| price_per_txn | 0.004536 |
| price_per_token | 0.00000131 |
| latency p50, p95, p99 | 1.51, 1.83, 1.86 |



The price performance comparison for different instance types is presented below. An interactive version of this chart is available [here](llama3-8b-trn1.html).

![Price performance comparison](business_summary.png)

### Latency Metrics Analysis

The following table provides token latency metrics including the overall latency, Time To First Token (TTFT), and Time Per Output Token (TPOT) for the `en_3000-3840` dataset.


_No token latency metrics data is available. To get token latency metrics enable streaming responses by setting `stream: True` in experiment configuration_.


### Failed experiments


There were a total of 9 experiment run(s) that failed at least one configured performance criteria: `Latency` < `2s`, `cost per 10k transactions`: `$100`, `error rate`: `0`. See table below.    
    



| experiment_name | payload_file | concurrency | error_rate_text | latency_p95_text | price_per_10k_txn_text |
|-----|-----|-----|-----|-----|-----|
| Meta-Llama-3-8B-Instruct-triton | payload_en_3000-3840.jsonl | 25 | <span style='color:green'>0.00</span> | <span style='color:red'>**4.28**</span> | <span style='color:green'>11.75</span> |
| Meta-Llama-3-8B-Instruct-triton | payload_en_3000-3840.jsonl | 18 | <span style='color:green'>0.00</span> | <span style='color:red'>**3.59**</span> | <span style='color:green'>12.27</span> |
| Meta-Llama-3-8B-Instruct-triton | payload_en_3000-3840.jsonl | 20 | <span style='color:green'>0.00</span> | <span style='color:red'>**4.36**</span> | <span style='color:green'>13.52</span> |
| Meta-Llama-3-8B-Instruct-triton | payload_en_3000-3840.jsonl | 15 | <span style='color:green'>0.00</span> | <span style='color:red'>**3.62**</span> | <span style='color:green'>15.12</span> |
| Meta-Llama-3-8B-Instruct-triton | payload_en_3000-3840.jsonl | 14 | <span style='color:green'>0.00</span> | <span style='color:red'>**3.32**</span> | <span style='color:green'>15.72</span> |
| Meta-Llama-3-8B-Instruct-triton | payload_en_3000-3840.jsonl | 12 | <span style='color:green'>0.00</span> | <span style='color:red'>**3.22**</span> | <span style='color:green'>17.15</span> |
| Meta-Llama-3-8B-Instruct-triton | payload_en_3000-3840.jsonl | 10 | <span style='color:green'>0.00</span> | <span style='color:red'>**3.10**</span> | <span style='color:green'>18.96</span> |
| Meta-Llama-3-8B-Instruct-triton | payload_en_3000-3840.jsonl | 8 | <span style='color:green'>0.00</span> | <span style='color:red'>**2.94**</span> | <span style='color:green'>21.46</span> |
| Meta-Llama-3-8B-Instruct-triton | payload_en_3000-3840.jsonl | 4 | <span style='color:green'>0.00</span> | <span style='color:red'>**2.25**</span> | <span style='color:green'>31.99</span> |



### Model evaluations

_Model evaluation data is not available_.

### Endpoint metrics

The following table provides endpoint utilization and invocation metrics.

_No endpoint metrics data is available_.

### Configuration

The configuration used for these tests is available in the [`config`](#configuration-file) file.

### Experiment cost

#### Model Benchmarking Cost

The cost to run each experiment is provided in the table below. The total cost for running all experiments is $7.52.


| experiment_name | instance_type | instance_count | duration_in_seconds | cost |
|-----|-----|-----|-----|-----|
| Meta-Llama-3-8B-Instruct-triton | trn1.32xlarge | nan | 1259.29 | 7.520752 |


#### Model Evaluation Cost

The cost to evaluate 1 candidate models using No LLM evaluators is 0.



The total cost incurred for **model benchmarking and evaluations**: $7.52.


## Per instance results

The following table provides the best combinations for running inference for different sizes prompts on different instance types. The following dataset(s) were used for this test: `2wikimqa_e.jsonl`, `2wikimqa.jsonl`, `hotpotqa_e.jsonl`, `hotpotqa.jsonl`, `narrativeqa.jsonl`, `triviaqa_e.jsonl`, `triviaqa.jsonl`.

|Dataset   | Instance type   | Recommendation   |
|---|---|---|
|`payload_en_1-500.jsonl`|`trn1.32xlarge`|The best option for staying within a latency budget of `2 seconds` on a `trn1.32xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 12`. A concurrency level of 12 achieves an `median latency of 1.92 seconds`, for an `average prompt size of 274 tokens` and `completion size of 40 tokens` with `364 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`trn1.32xlarge`|The best option for staying within a latency budget of `2 seconds` on a `trn1.32xlarge` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `median latency of 1.62 seconds`, for an `average prompt size of 1556 tokens` and `completion size of 47 tokens` with `66 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`trn1.32xlarge`|The best option for staying within a latency budget of `2 seconds` on a `trn1.32xlarge` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `median latency of 1.87 seconds`, for an `average prompt size of 2526 tokens` and `completion size of 53 tokens` with `40 transactions/minute`.|
|`payload_en_3000-3840.jsonl`|`trn1.32xlarge`|The best option for staying within a latency budget of `2 seconds` on a `trn1.32xlarge` for the `payload_en_3000-3840.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `median latency of 1.51 seconds`, for an `average prompt size of 3433 tokens` and `completion size of 42 tokens` with `79 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`trn1.32xlarge`|The best option for staying within a latency budget of `2 seconds` on a `trn1.32xlarge` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `median latency of 1.98 seconds`, for an `average prompt size of 827 tokens` and `completion size of 47 tokens` with `237 transactions/minute`.|

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
  sagemaker_execution_role: arn:aws:iam::988564344122:user/madhureks
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
  - 15
  - 18
  - 20
  - 25
  deploy: true
  deployment_script: ec2_deploy.py
  ec2:
    model_loading_timeout: 8000
  env: null
  ep_name: http://localhost:8080/v2/models/Meta-Llama-3-8B-Instruct/generate
  image_uri: tritonserver-neuronx:fmbench
  inference_script: ec2_predictor.py
  inference_spec:
    backend: djl
    container_type: triton
    inference_container_params:
      amp: f16
      attention_layout: BSH
      collectives_layout: BSH
      context_length_estimate: 3072, 3584, 4096
      max_model_len: 4096
      max_rolling_batch_size: 64
      model_loader: tnx
      model_loading_timeout: 2400
      n_positions: 4096
      output_formatter: json
      rolling_batch: auto
      rolling_batch_strategy: continuous_batching
      serving.properties: null
      tp_degree: 8
      trust_remote_code: true
    model_copies: max
    parameter_set: ec2_djl
    parameters:
      max_tokens: 100
      top_k: 50
    shm_size: 12g
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
  read_bucket: sagemaker-fmbench-read-us-west-2-988564344122
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
