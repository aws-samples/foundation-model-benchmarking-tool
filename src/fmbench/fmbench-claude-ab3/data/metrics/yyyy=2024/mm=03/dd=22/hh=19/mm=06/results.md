
# Results for performance benchmarking

**Last modified (UTC): 2024-03-22 23:41:52.159969**

## Summary

We did performance benchmarking for the `claude` model on "`ClaudeInstant-ODT`" instance on multiple datasets and based on the test results the best price performance for dataset `en_1000-2000` is provided by the `ClaudeInstant-ODT` instance type.  
| Information | Value |
|-----|-----|
| experiment_name | claude-instant-v1 |
| payload_file | payload_en_1000-2000.jsonl |
| instance_type | ClaudeInstant-ODT |
| concurrency | 1 |
| error_rate | 0.0 |
| prompt_token_count_mean | 1623 |
| prompt_token_throughput | 764 |
| completion_token_count_mean | 58 |
| completion_token_throughput | 23 |
| latency_mean | 2.54 |
| transactions_per_minute | 28 |
| price_per_hour | [{'input-per-1k-tokens': 0.0008}, {'output-per-1k-tokens': 0.0024}] |
| price_per_txn | 0.001438 |


The price performance comparison for different instance types is presented below:

![Price performance comparison](business_summary.png)

The configuration used for these tests is available in the [`config`](config-claude-models.yml) file.

The cost to run each experiment is provided in the table below. The total cost for running all experiments is $0.00.

| experiment_name | instance_type | duration_in_seconds | cost |
|-----|-----|-----|-----|
| claude-instant-v1 | ClaudeInstant-ODT | 46.71 | 0.0 |




## Per instance results

The following table provides the best combinations for running inference for different sizes prompts on different instance types.

|Dataset   | Instance type   | Recommendation   |
|---|---|---|
|`payload_en_1000-2000.jsonl`|`ClaudeInstant-ODT`|The best option for staying within a latency budget of `20 seconds` on a `ClaudeInstant-ODT` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 2.54 seconds`, for an `average prompt size of 1623 tokens` and `completion size of 58 tokens` with `28 transactions/minute`.|

## Plots

The following plots provide insights into the results from the different experiments run.

![Error rates for different concurrency levels and instance types](error_rates.png)

![Tokens vs latency for different concurrency levels and instance types](tokens_vs_latency.png)

![Concurrency Vs latency for different instance type for selected dataset](concurrency_vs_inference_latency.png)
