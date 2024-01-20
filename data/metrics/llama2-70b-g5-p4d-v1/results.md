
# Results for performance benchmarking

**Last modified (UTC): 2024-01-20 16:19:02.241122**

## Summary

We did performance benchmarking for the `Llama2-70b` model on "`ml.p4d.24xlarge`, `ml.g5.48xlarge`" instances on multiple datasets and based on the test results the best price performance for dataset `en_3000-4000` is provided by the `ml.p4d.24xlarge` instance type.  
| Information | Value |
|-----|-----|
| experiment_name | llama2-13b-p4d.24xlarge-tgi-inference-2.0.1-tgi0.9.3-gpu-py39-cu118 |
| payload_file | payload_en_3000-4000.jsonl |
| instance_type | ml.p4d.24xlarge |
| concurrency | 2 |
| error_rate | 0.0 |
| prompt_token_count_mean | 3482 |
| prompt_token_throughput | 1680 |
| completion_token_count_mean | 3507 |
| completion_token_throughput | 1690 |
| latency_mean | 3.97 |
| transactions_per_minute | 28 |
| price_per_hour | 37.688 |
| price_per_txn | 0.0224 |


The price performance comparison for different instance types is presented below:

![Price performance comparison](business_summary.png)

The configuration used for these tests is available in the [`config`](config-llama2-70b-g5-p4d.yml) file.


## Per instance results

The following table provides the best combinations for running inference for different sizes prompts on different instance types.
|Dataset   | Instance type   | Recommendation   |
|---|---|---|
|`payload_en_1-500.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.p4d.24xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 1.38 seconds`, for an `average prompt size of 304 tokens` and `completion size of 328 tokens` with `43 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.p4d.24xlarge` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 4`. A concurrency level of 4 achieves an `average latency of 4.24 seconds`, for an `average prompt size of 1548 tokens` and `completion size of 1584 tokens` with `38 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.p4d.24xlarge` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 4`. A concurrency level of 4 achieves an `average latency of 5.53 seconds`, for an `average prompt size of 2503 tokens` and `completion size of 2537 tokens` with `32 transactions/minute`.|
|`payload_en_3000-4000.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.p4d.24xlarge` for the `payload_en_3000-4000.jsonl` dataset is a `concurrency level of 4`. A concurrency level of 4 achieves an `average latency of 6.34 seconds`, for an `average prompt size of 3482 tokens` and `completion size of 3507 tokens` with `28 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.p4d.24xlarge` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 1.56 seconds`, for an `average prompt size of 980 tokens` and `completion size of 1004 tokens` with `38 transactions/minute`.|
|`payload_en_1-500.jsonl`|`ml.g5.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.48xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 6.18 seconds`, for an `average prompt size of 304 tokens` and `completion size of 102 tokens` with `9 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`ml.g5.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.48xlarge` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 14.51 seconds`, for an `average prompt size of 1600 tokens` and `completion size of 75 tokens` with `7 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`ml.g5.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.48xlarge` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 19.64 seconds`, for an `average prompt size of 2503 tokens` and `completion size of 69 tokens` with `5 transactions/minute`.|
|`payload_en_3000-4000.jsonl`|`ml.g5.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.48xlarge` for the `payload_en_3000-4000.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 15.22 seconds`, for an `average prompt size of 3478 tokens` and `completion size of 65 tokens` with `3 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`ml.g5.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.48xlarge` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 8.39 seconds`, for an `average prompt size of 980 tokens` and `completion size of 102 tokens` with `7 transactions/minute`.|

## Plots

The following plots provide insights into the results from the different experiments run.

![Error rates for different concurrency levels and instance types](error_rates.png)

![Tokens vs latency for different concurrency levels and instance types](tokens_vs_latency.png)

![Concurrency Vs latency for different instance type for selected dataset](concurrency_vs_inference_latency.png)
