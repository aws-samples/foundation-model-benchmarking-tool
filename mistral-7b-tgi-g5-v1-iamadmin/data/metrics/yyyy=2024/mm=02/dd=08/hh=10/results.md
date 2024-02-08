
# Results for performance benchmarking

**Last modified (UTC): 2024-02-08 15:44:19.728203**

## Summary

We did performance benchmarking for the `mistral7b` model on "`ml.g5.2xlarge`" instance on multiple datasets and based on the test results the best price performance for dataset `en_1-500` is provided by the `ml.g5.2xlarge` instance type.  
| Information | Value |
|-----|-----|
| experiment_name | mistral-7b-g5-huggingface-pytorch-tgi-inference-2.0.1-tgi1.1.0 |
| payload_file | payload_en_1-500.jsonl |
| instance_type | ml.g5.2xlarge |
| concurrency | 1 |
| error_rate | 0.0 |
| prompt_token_count_mean | 304 |
| prompt_token_throughput | 96 |
| completion_token_count_mean | 82 |
| completion_token_throughput | 23 |
| latency_mean | 3.37 |
| transactions_per_minute | 18 |
| price_per_hour | 1.515 |
| price_per_txn | 0.0014 |


The price performance comparison for different instance types is presented below:

![Price performance comparison](business_summary.png)

The configuration used for these tests is available in the [`config`](config-mistral-7b-tgi-g5.yml) file.


## Per instance results

The following table provides the best combinations for running inference for different sizes prompts on different instance types.
|Dataset   | Instance type   | Recommendation   |
|---|---|---|
|`payload_en_1-500.jsonl`|`ml.g5.2xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.2xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 3.37 seconds`, for an `average prompt size of 304 tokens` and `completion size of 82 tokens` with `18 transactions/minute`.|

## Plots

The following plots provide insights into the results from the different experiments run.

![Error rates for different concurrency levels and instance types](error_rates.png)

![Tokens vs latency for different concurrency levels and instance types](tokens_vs_latency.png)

![Concurrency Vs latency for different instance type for selected dataset](concurrency_vs_inference_latency.png)
