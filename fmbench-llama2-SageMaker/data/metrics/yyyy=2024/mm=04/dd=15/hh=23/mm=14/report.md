
# Results for performance benchmarking

**Last modified (UTC): 2024-04-15 23:48:23.870126**

## Summary

We did performance benchmarking for `Llama2 on Bedrock & SageMaker` on "`ml.p4d.24xlarge`, `meta.llama2-13b-chat-v1`" on multiple datasets and based on the test results the best price performance for dataset `en_3000-3840` is provided by the `meta.llama2-13b-chat-v1`.


| Information | Value |
|-----|-----|
| experiment_name | meta.llama2-13b-chat-v1 |
| payload_file | payload_en_3000-3840.jsonl |
| instance_type | meta.llama2-13b-chat-v1 |
| concurrency | 10 |
| error_rate | 0.0 |
| prompt_token_count_mean | 2923 |
| prompt_token_throughput | 9296 |
| completion_token_count_mean | 24 |
| completion_token_throughput | 72 |
| latency_mean | 1.48 |
| latency_p50 | 1.28 |
| latency_p95 | 2.51 |
| latency_p99 | 2.86 |
| transactions_per_minute | 190 |
| price_per_txn | 0.002216 |


The price performance comparison for different instance types is presented below:

![Price performance comparison](business_summary.png)

There were a total of 6 experiment run(s) that failed at least one configured performance criteria: `Latency` < `2s`, `cost per 10k transactions`: `$50`, `error rate`: `0`. See table below.    
    

| experiment_name | payload_file | concurrency | error_rate_text | latency_mean_text | price_per_10k_txn_text |
|-----|-----|-----|-----|-----|-----|
| llama2-13b-p4d.24xlarge-trt-inference | payload_en_3000-3840.jsonl | 1 | <span style='color:red'>**0.06**</span> | <span style='color:green'>0.49</span> | <span style='color:green'>43.32</span> |
| llama2-13b-p4d.24xlarge-trt-inference | payload_en_3000-3840.jsonl | 2 | <span style='color:red'>**0.06**</span> | <span style='color:green'>0.46</span> | <span style='color:green'>25.85</span> |
| llama2-13b-p4d.24xlarge-trt-inference | payload_en_3000-3840.jsonl | 4 | <span style='color:red'>**0.08**</span> | <span style='color:green'>0.67</span> | <span style='color:green'>24.93</span> |
| llama2-13b-p4d.24xlarge-trt-inference | payload_en_3000-3840.jsonl | 6 | <span style='color:red'>**0.12**</span> | <span style='color:green'>0.80</span> | <span style='color:green'>21.73</span> |
| llama2-13b-p4d.24xlarge-trt-inference | payload_en_3000-3840.jsonl | 8 | <span style='color:red'>**0.12**</span> | <span style='color:green'>0.92</span> | <span style='color:green'>16.80</span> |
| llama2-13b-p4d.24xlarge-trt-inference | payload_en_3000-3840.jsonl | 10 | <span style='color:red'>**0.20**</span> | <span style='color:green'>0.98</span> | <span style='color:green'>18.15</span> |


The configuration used for these tests is available in the [`config`](config-bedrock-sagemaker-llama2.yml) file.

The cost to run each experiment is provided in the table below. The total cost for running all experiments is $3.90.



| experiment_name | instance_type | duration_in_seconds | cost |
|-----|-----|-----|-----|
| llama2-13b-p4d.24xlarge-trt-inference | ml.p4d.24xlarge | 263.55 | 2.75905 |
| meta.llama2-13b-chat-v1 | meta.llama2-13b-chat-v1 | 536.69 | 1.143778 |




## Per instance results

The following table provides the best combinations for running inference for different sizes prompts on different instance types. The following dataset(s) were used for this test: `2wikimqa_e.jsonl`, `2wikimqa.jsonl`, `hotpotqa_e.jsonl`, `hotpotqa.jsonl`, `narrativeqa.jsonl`, `triviaqa_e.jsonl`, `triviaqa.jsonl`.

|Dataset   | Instance type   | Recommendation   |
|---|---|---|
|`payload_en_1-500.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `2 seconds` on a `ml.p4d.24xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 10`. A concurrency level of 10 achieves an `average latency of 0.36 seconds`, for an `average prompt size of 309 tokens` and `completion size of 26 tokens` with `921 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `2 seconds` on a `ml.p4d.24xlarge` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 10`. A concurrency level of 10 achieves an `average latency of 0.8 seconds`, for an `average prompt size of 1632 tokens` and `completion size of 42 tokens` with `410 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `2 seconds` on a `ml.p4d.24xlarge` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 10`. A concurrency level of 10 achieves an `average latency of 1.1 seconds`, for an `average prompt size of 2516 tokens` and `completion size of 53 tokens` with `322 transactions/minute`.|
|`payload_en_3000-3840.jsonl`|`ml.p4d.24xlarge`|This experiment did not find any combination of concurrency level and other configuration settings that could provide a response within a latency budget of `2 seconds` on a `ml.p4d.24xlarge` for the `payload_en_3000-3840.jsonl` dataset.|
|`payload_en_500-1000.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `2 seconds` on a `ml.p4d.24xlarge` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 10`. A concurrency level of 10 achieves an `average latency of 0.34 seconds`, for an `average prompt size of 969 tokens` and `completion size of 16 tokens` with `1016 transactions/minute`.|
|`payload_en_1-500.jsonl`|`meta.llama2-13b-chat-v1`|The best option for staying within a latency budget of `2 seconds` on a `meta.llama2-13b-chat-v1` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 10`. A concurrency level of 10 achieves an `average latency of 0.79 seconds`, for an `average prompt size of 267 tokens` and `completion size of 24 tokens` with `382 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`meta.llama2-13b-chat-v1`|The best option for staying within a latency budget of `2 seconds` on a `meta.llama2-13b-chat-v1` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 10`. A concurrency level of 10 achieves an `average latency of 1.4 seconds`, for an `average prompt size of 1394 tokens` and `completion size of 31 tokens` with `219 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`meta.llama2-13b-chat-v1`|The best option for staying within a latency budget of `2 seconds` on a `meta.llama2-13b-chat-v1` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 10`. A concurrency level of 10 achieves an `average latency of 1.9 seconds`, for an `average prompt size of 2157 tokens` and `completion size of 43 tokens` with `165 transactions/minute`.|
|`payload_en_3000-3840.jsonl`|`meta.llama2-13b-chat-v1`|The best option for staying within a latency budget of `2 seconds` on a `meta.llama2-13b-chat-v1` for the `payload_en_3000-3840.jsonl` dataset is a `concurrency level of 10`. A concurrency level of 10 achieves an `average latency of 1.48 seconds`, for an `average prompt size of 2923 tokens` and `completion size of 24 tokens` with `190 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`meta.llama2-13b-chat-v1`|The best option for staying within a latency budget of `2 seconds` on a `meta.llama2-13b-chat-v1` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 10`. A concurrency level of 10 achieves an `average latency of 0.63 seconds`, for an `average prompt size of 818 tokens` and `completion size of 14 tokens` with `442 transactions/minute`.|

## Plots

The following plots provide insights into the results from the different experiments run.

![Error rates for different concurrency levels and instance types](error_rates.png)

![Tokens vs latency for different concurrency levels and instance types](tokens_vs_latency.png)

![Concurrency Vs latency for different instance type for selected dataset](concurrency_vs_inference_latency.png)
