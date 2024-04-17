
# Results for performance benchmarking

**Last modified (UTC): 2024-04-16 20:15:41.665026**

## Summary

We did performance benchmarking for `Llama2-7b` on "`ml.inf2.48xlarge`" on multiple datasets and based on the test results the best price performance for dataset `en_1-924` is provided by the `ml.inf2.48xlarge`.


| Information | Value |
|-----|-----|
| experiment_name | llama2-7b-g5.2xlarge-bring-your-own-sm-endpoint |
| payload_file | payload_en_1-924.jsonl |
| instance_type | ml.inf2.48xlarge |
| concurrency | 4 |
| error_rate | 0.0 |
| prompt_token_count_mean | 304 |
| prompt_token_throughput | 418 |
| completion_token_count_mean | 329 |
| completion_token_throughput | 453 |
| latency_mean | 2.86 |
| latency_p50 | 2.85 |
| latency_p95 | 2.9 |
| latency_p99 | 2.9 |
| transactions_per_minute | 82 |
| price_per_txn | 0.003167 |


The price performance comparison for different instance types is presented below:

![Price performance comparison](business_summary.png)

There were a total of 2 experiment run(s) that failed at least one configured performance criteria: `Latency` < `5s`, `cost per 10k transactions`: `$50`, `error rate`: `0`. See table below.    
    

| experiment_name | payload_file | concurrency | error_rate_text | latency_mean_text | price_per_10k_txn_text |
|-----|-----|-----|-----|-----|-----|
| llama2-7b-g5.2xlarge-bring-your-own-sm-endpoint | payload_en_1-924.jsonl | 1 | <span style='color:green'>0.00</span> | <span style='color:green'>2.02</span> | <span style='color:red'>**89.54**</span> |
| llama2-7b-g5.2xlarge-bring-your-own-sm-endpoint | payload_en_1-924.jsonl | 2 | <span style='color:green'>0.00</span> | <span style='color:green'>2.80</span> | <span style='color:red'>**61.83**</span> |


The configuration used for these tests is available in the [`config`](config-llama2-7b-byo-sagemaker-endpoint.yml) file.

The cost to run each experiment is provided in the table below. The total cost for running all experiments is $0.04.



| experiment_name | instance_type | duration_in_seconds | cost |
|-----|-----|-----|-----|
| llama2-7b-g5.2xlarge-bring-your-own-sm-endpoint | ml.inf2.48xlarge | 8.71 | 0.037715 |




## Per instance results

The following table provides the best combinations for running inference for different sizes prompts on different instance types. The following dataset(s) were used for this test: `2wikimqa_e.jsonl`, `2wikimqa.jsonl`, `hotpotqa_e.jsonl`, `hotpotqa.jsonl`, `narrativeqa.jsonl`, `triviaqa_e.jsonl`, `triviaqa.jsonl`.

|Dataset   | Instance type   | Recommendation   |
|---|---|---|
|`payload_en_1-924.jsonl`|`ml.inf2.48xlarge`|The best option for staying within a latency budget of `5 seconds` on a `ml.inf2.48xlarge` for the `payload_en_1-924.jsonl` dataset is a `concurrency level of 4`. A concurrency level of 4 achieves an `average latency of 2.86 seconds`, for an `average prompt size of 304 tokens` and `completion size of 329 tokens` with `82 transactions/minute`.|

## Plots

The following plots provide insights into the results from the different experiments run.

![Error rates for different concurrency levels and instance types](error_rates.png)

![Tokens vs latency for different concurrency levels and instance types](tokens_vs_latency.png)

![Concurrency Vs latency for different instance type for selected dataset](concurrency_vs_inference_latency.png)
