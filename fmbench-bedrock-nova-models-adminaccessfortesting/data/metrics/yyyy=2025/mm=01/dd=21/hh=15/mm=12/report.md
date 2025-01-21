
# Results for performance benchmarking Nova models available in Amazon Bedrock

|**Last modified (UTC)** | **FMBench version**  |
|---|---|
|2025-01-21 15:25:03.456869|2.0.26|


## Summary

We did performance benchmarking for `Nova models available in Amazon Bedrock` on "`us.amazon.nova-micro-v1:0`" on multiple datasets and based on the test results the best price performance for dataset `en_500-1000` is provided by the `us.amazon.nova-micro-v1:0`.


| Information | Value |
|-----|-----|
| experiment_name | us.amazon.nova-micro-v1:0 |
| payload_file | payload_en_500-1000.jsonl |
| instance_type | us.amazon.nova-micro-v1:0 |
| instance_count | 1.0 |
| concurrency | 2 |
| error_rate | 0.0 |
| prompt_token_count_mean | 1121 |
| prompt_token_throughput | 3335 |
| completion_token_count_mean | 48 |
| completion_token_throughput | 143 |
| transactions_per_minute | 178 |
| price_per_txn | 4.6e-05 |
| price_per_token | 0.00000004 |
| latency p50, p95, p99 | 0.61, 0.64, 0.64 |



The price performance comparison for different instance types is presented below. An interactive version of this chart is available [here](business_summary.html).

![Price performance comparison](business_summary.png)

### Latency Metrics Analysis

The following table provides token latency metrics including the overall latency, Time To First Token (TTFT), and Time Per Output Token (TPOT) for the `en_500-1000` dataset.


_No token latency metrics data is available. To get token latency metrics enable streaming responses by setting `stream: True` in experiment configuration_.


### Failed experiments


All experiments satisfied the configured performance criteria: `Latency` < `2s`, `cost per 10k transactions`: `$20`, `error rate`: `0`.






### Model evaluations


Model evaluations were performed by a panel of 3 LLM judges: anthropic.claude-3-5-sonnet-20241022-v2:0, cohere.command-r-plus-v1:0, us.meta.llama3-3-70b-instruct-v1:0. Model outputs were compared with ground truth available in the dataset by the judge models. The following charts provide the results of model evaluations.

| candidate_model | judge_anthropic.claude-3-5-sonnet-20241022-v2:0_accuracy | judge_cohere.command-r-plus-v1:0_accuracy | judge_us.meta.llama3-3-70b-instruct-v1:0_accuracy | majority_voting_accuracy |
|-----|-----|-----|-----|-----|
| us.amazon.nova-micro-v1:0 | 87.5% | 87.5% | 87.5% | 87.5% |


![Overall model accuracy](overall_candidate_model_majority_voting_accuracy.png)
View an interactive version of the overall accuracy chart [here](overall_candidate_model_majority_voting_accuracy.html)

![Model accuracy trend across prompt sizes](accuracy_trajectory_per_payload.png)
View an interactive version of the accuracy trajectory chart [here](accuracy_trajectory_per_payload.html)



### Endpoint metrics

The following table provides endpoint utilization and invocation metrics.

_No endpoint metrics data is available_.

### Configuration

The configuration used for these tests is available in the [`config`](#configuration-file) file.

### Experiment cost

#### Model Benchmarking Cost

The cost to run each experiment is provided in the table below. The total cost for running all experiments is $0.00.


| experiment_name | instance_type | instance_count | duration_in_seconds | cost |
|-----|-----|-----|-----|-----|
| us.amazon.nova-micro-v1:0 | us.amazon.nova-micro-v1:0 | nan | 12.91 | 0.000912 |


#### Model Evaluation Cost

The cost to evaluate 1 candidate models using 3 LLM evaluators is $0.0172.

| judge_model_id | total_cost | prompt_token_count | completion_token_count |
|-----|-----|-----|-----|
| cohere.command-r-plus-v1:0 | 0.0147 | 3175 | 346 |
| us.meta.llama3-3-70b-instruct-v1:0 | 0.0025 | 3189 | 307 |
| anthropic.claude-3-5-sonnet-20241022-v2:0 | 0.0 | 3757 | 515 |


The total cost incurred for **model benchmarking and evaluations**: $0.0181.


## Per instance results

The following table provides the best combinations for running inference for different sizes prompts on different instance types. The following dataset(s) were used for this test: `2wikimqa_e.jsonl`, `2wikimqa.jsonl`, `hotpotqa_e.jsonl`, `hotpotqa.jsonl`, `narrativeqa.jsonl`, `triviaqa_e.jsonl`, `triviaqa.jsonl`.

|Dataset   | Instance type   | Recommendation   |
|---|---|---|
|`payload_en_1-500.jsonl`|`us.amazon.nova-micro-v1:0`|The best option for staying within a latency budget of `2 seconds` on a `us.amazon.nova-micro-v1:0` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `median latency of 0.53 seconds`, for an `average prompt size of 311 tokens` and `completion size of 34 tokens` with `205 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`us.amazon.nova-micro-v1:0`|The best option for staying within a latency budget of `2 seconds` on a `us.amazon.nova-micro-v1:0` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `median latency of 0.61 seconds`, for an `average prompt size of 1121 tokens` and `completion size of 48 tokens` with `178 transactions/minute`.|

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
  sagemaker_execution_role: arn:aws:iam::218208277580:role/adminaccessfortesting
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
    payload_file: payload_en_3000-3840.jsonl
  ground_truth_col_key: answers
  prompt_template_keys:
  - input
  - context
  question_col_key: input
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
  env: null
  ep_name: us.amazon.nova-micro-v1:0
  inference_script: bedrock_predictor.py
  inference_spec:
    parameter_set: bedrock
    parameters:
      caching: false
      max_tokens: 100
      temperature: 0.1
      top_p: 0.92
      use_boto3: true
    split_input_and_parameters: false
  instance_count: null
  instance_type: us.amazon.nova-micro-v1:0
  model_name: us.amazon.nova-micro-v1:0
  name: us.amazon.nova-micro-v1:0
  payload_files:
  - payload_en_1-500.jsonl
  - payload_en_500-1000.jsonl
general:
  model_name: Nova models available in Amazon Bedrock
  name: fmbench-bedrock-nova-models
inference_parameters:
  bedrock:
    caching: false
    max_tokens: 100
    temperature: 0.1
    top_p: 0.92
    use_boto3: true
metrics:
  dataset_of_interest: en_500-1000
model_evaluations: model_eval_all_info.yml
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
  2_deploy_model.ipynb: false
  3_run_inference.ipynb: false
  4_get_evaluations.ipynb: true
  5_model_metric_analysis.ipynb: true
  6_cleanup.ipynb: false
s3_read_data:
  config_files:
  - pricing.yml
  configs_prefix: configs
  local_file_system_path: /tmp/fmbench-read
  prompt_template_dir: prompt_template
  prompt_template_file: prompt_template_nova.txt
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
  tokenizer_prefix: tokenizer

```
