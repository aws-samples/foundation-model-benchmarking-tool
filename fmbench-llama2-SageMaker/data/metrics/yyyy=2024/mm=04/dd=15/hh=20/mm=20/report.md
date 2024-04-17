
# Results for performance benchmarking

**Last modified (UTC): 2024-04-15 20:33:15.922191**

## Summary

We did performance benchmarking for `Llama2 on Bedrock & SageMaker` on "`ml.p4d.24xlarge`, `meta.llama2-13b-chat-v1`" on multiple datasets and based on the test results the best price performance for dataset `en_500-1000` is provided by the `ml.p4d.24xlarge`.


| Information | Value |
|-----|-----|
| experiment_name | llama2-13b-p4d.24xlarge-trt-inference |
| payload_file | payload_en_500-1000.jsonl |
| instance_type | ml.p4d.24xlarge |
| concurrency | 4 |
| error_rate | 0.0 |
| prompt_token_count_mean | 969 |
| prompt_token_throughput | 13591 |
| completion_token_count_mean | 16 |
| completion_token_throughput | 224 |
| latency_mean | 0.27 |
| latency_p50 | 0.27 |
| latency_p95 | 0.28 |
| latency_p99 | 0.28 |
| transactions_per_minute | 841 |
| price_per_txn | 0.000747 |


The price performance comparison for different instance types is presented below:

![Price performance comparison](business_summary.png)

There were a total of 1 experiment run(s) that failed at least one configured performance criteria: `Latency` < `2s`, `cost per 10k transactions`: `$20`, `error rate`: `0`. See table below.    
    

| experiment_name | payload_file | concurrency | error_rate_text | latency_mean_text | price_per_10k_txn_text |
|-----|-----|-----|-----|-----|-----|
| llama2-13b-p4d.24xlarge-trt-inference | payload_en_500-1000.jsonl | 1 | <span style='color:green'>0.00</span> | <span style='color:green'>1.34</span> | <span style='color:red'>**142.76**</span> |


The configuration used for these tests is available in the [`config`](config-bedrock-sagemaker-llama2.yml) file.

The cost to run each experiment is provided in the table below. The total cost for running all experiments is $0.07.



| experiment_name | instance_type | duration_in_seconds | cost |
|-----|-----|-----|-----|
| llama2-13b-p4d.24xlarge-trt-inference | ml.p4d.24xlarge | 6.22 | 0.065161 |
| meta.llama2-13b-chat-v1 | meta.llama2-13b-chat-v1 | 9.0 | 0.005963 |




## Per instance results

The following table provides the best combinations for running inference for different sizes prompts on different instance types. The following dataset(s) were used for this test: `2wikimqa_e.jsonl`, `2wikimqa.jsonl`, `hotpotqa_e.jsonl`, `hotpotqa.jsonl`, `narrativeqa.jsonl`, `triviaqa_e.jsonl`, `triviaqa.jsonl`.

|Dataset   | Instance type   | Recommendation   |
|---|---|---|
|`payload_en_1-500.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `2 seconds` on a `ml.p4d.24xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 4`. A concurrency level of 4 achieves an `average latency of 0.37 seconds`, for an `average prompt size of 309 tokens` and `completion size of 27 tokens` with `561 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `2 seconds` on a `ml.p4d.24xlarge` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 4`. A concurrency level of 4 achieves an `average latency of 0.27 seconds`, for an `average prompt size of 969 tokens` and `completion size of 16 tokens` with `841 transactions/minute`.|
|`payload_en_1-500.jsonl`|`meta.llama2-13b-chat-v1`|The best option for staying within a latency budget of `2 seconds` on a `meta.llama2-13b-chat-v1` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 4`. A concurrency level of 4 achieves an `average latency of 1.05 seconds`, for an `average prompt size of 267 tokens` and `completion size of 23 tokens` with `206 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`meta.llama2-13b-chat-v1`|The best option for staying within a latency budget of `2 seconds` on a `meta.llama2-13b-chat-v1` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 4`. A concurrency level of 4 achieves an `average latency of 0.94 seconds`, for an `average prompt size of 818 tokens` and `completion size of 14 tokens` with `239 transactions/minute`.|

## Plots

The following plots provide insights into the results from the different experiments run.

![Error rates for different concurrency levels and instance types](error_rates.png)

![Tokens vs latency for different concurrency levels and instance types](tokens_vs_latency.png)

![Concurrency Vs latency for different instance type for selected dataset](concurrency_vs_inference_latency.png)
