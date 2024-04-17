
# Results for performance benchmarking

**Last modified (UTC): 2024-04-15 18:27:16.123921**

## Summary

We did performance benchmarking for `Claude Sonnet available in Amazon Bedrock` on "`anthropic.claude-3-sonnet-20240229-v1:0`, `anthropic.claude-v3-sonnet-pt-nc`" on multiple datasets and based on the test results the best price performance for dataset `en_1000-2000` is provided by the `anthropic.claude-v3-sonnet-pt-nc`.


| Information | Value |
|-----|-----|
| experiment_name | anthropic.claude-v3-sonnet-pt |
| payload_file | payload_en_1000-2000.jsonl |
| instance_type | anthropic.claude-v3-sonnet-pt-nc |
| concurrency | 2 |
| error_rate | 0.0 |
| prompt_token_count_mean | 1545 |
| prompt_token_throughput | 1413 |
| completion_token_count_mean | 52 |
| completion_token_throughput | 46 |
| latency_mean | 2.02 |
| latency_p50 | 2.02 |
| latency_p95 | 2.23 |
| latency_p99 | 2.25 |
| transactions_per_minute | 55 |
| price_per_txn | 0.026667 |


The price performance comparison for different instance types is presented below:

![Price performance comparison](business_summary.png)

There were a total of 1 experiment run(s) that failed at least one configured performance criteria: `Latency` < `5s`, `cost per 10k transactions`: `$350`, `error rate`: `0`. See table below.    
    

| experiment_name | payload_file | concurrency | error_rate_text | latency_mean_text | price_per_10k_txn_text |
|-----|-----|-----|-----|-----|-----|
| anthropic.claude-v3-sonnet-pt | payload_en_1000-2000.jsonl | 1 | <span style='color:green'>0.00</span> | <span style='color:green'>2.18</span> | <span style='color:red'>**505.75**</span> |


The configuration used for these tests is available in the [`config`](config-bedrock-claude.yml) file.

The cost to run each experiment is provided in the table below. The total cost for running all experiments is $88.18.



| experiment_name | instance_type | duration_in_seconds | cost |
|-----|-----|-----|-----|
| anthropic.claude-v3-sonnet-pt | anthropic.claude-v3-sonnet-pt-nc | 63.99 | 88.0 |
| anthropic.claude-3-sonnet-20240229-v1:0 | anthropic.claude-3-sonnet-20240229-v1:0 | 86.65 | 0.183285 |




## Per instance results

The following table provides the best combinations for running inference for different sizes prompts on different instance types. The following dataset(s) were used for this test: `2wikimqa_e.jsonl`, `2wikimqa.jsonl`, `hotpotqa_e.jsonl`, `hotpotqa.jsonl`, `narrativeqa.jsonl`, `triviaqa_e.jsonl`, `triviaqa.jsonl`.

|Dataset   | Instance type   | Recommendation   |
|---|---|---|
|`payload_en_1-500.jsonl`|`anthropic.claude-3-sonnet-20240229-v1:0`|The best option for staying within a latency budget of `5 seconds` on a `anthropic.claude-3-sonnet-20240229-v1:0` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.17 seconds`, for an `average prompt size of 276 tokens` and `completion size of 31 tokens` with `96 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`anthropic.claude-3-sonnet-20240229-v1:0`|The best option for staying within a latency budget of `5 seconds` on a `anthropic.claude-3-sonnet-20240229-v1:0` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 2.35 seconds`, for an `average prompt size of 1545 tokens` and `completion size of 54 tokens` with `40 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`anthropic.claude-3-sonnet-20240229-v1:0`|The best option for staying within a latency budget of `5 seconds` on a `anthropic.claude-3-sonnet-20240229-v1:0` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 3.2 seconds`, for an `average prompt size of 891 tokens` and `completion size of 62 tokens` with `26 transactions/minute`.|
|`payload_en_1-500.jsonl`|`anthropic.claude-v3-sonnet-pt-nc`|The best option for staying within a latency budget of `5 seconds` on a `anthropic.claude-v3-sonnet-pt-nc` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.44 seconds`, for an `average prompt size of 276 tokens` and `completion size of 31 tokens` with `82 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`anthropic.claude-v3-sonnet-pt-nc`|The best option for staying within a latency budget of `5 seconds` on a `anthropic.claude-v3-sonnet-pt-nc` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 2.02 seconds`, for an `average prompt size of 1545 tokens` and `completion size of 52 tokens` with `55 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`anthropic.claude-v3-sonnet-pt-nc`|The best option for staying within a latency budget of `5 seconds` on a `anthropic.claude-v3-sonnet-pt-nc` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 2.0 seconds`, for an `average prompt size of 891 tokens` and `completion size of 62 tokens` with `57 transactions/minute`.|

## Plots

The following plots provide insights into the results from the different experiments run.

![Error rates for different concurrency levels and instance types](error_rates.png)

![Tokens vs latency for different concurrency levels and instance types](tokens_vs_latency.png)

![Concurrency Vs latency for different instance type for selected dataset](concurrency_vs_inference_latency.png)
