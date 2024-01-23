
# Results for performance benchmarking

**Last modified (UTC): 2024-01-23 22:38:07.921983**

## Summary

We did performance benchmarking for the `Llama2-70b` model on "`ml.g5.48xlarge`, `ml.p4d.24xlarge`" instances on multiple datasets and based on the test results the best price performance for dataset `en_3000-4000` is provided by the `ml.p4d.24xlarge` instance type.  
| Information | Value |
|-----|-----|
| experiment_name | llama2-70b-p4d.24xlarge-tgi-inference-2.0.1-tgi0.9.3-gpu-py39-cu118 |
| payload_file | payload_en_3000-4000.jsonl |
| instance_type | ml.p4d.24xlarge |
| concurrency | 6 |
| error_rate | 0.0 |
| prompt_token_count_mean | 3454 |
| prompt_token_throughput | 1770 |
| completion_token_count_mean | 3480 |
| completion_token_throughput | 1782 |
| latency_mean | 8.04 |
| transactions_per_minute | 30 |
| price_per_hour | 37.688 |
| price_per_txn | 0.0209 |


The price performance comparison for different instance types is presented below:

![Price performance comparison](business_summary.png)

The configuration used for these tests is available in the [`config`](config-llama2-70b-g5-p4d-tgi.yml) file.


## Per instance results

The following table provides the best combinations for running inference for different sizes prompts on different instance types.
|Dataset   | Instance type   | Recommendation   |
|---|---|---|
|`payload_en_1-500.jsonl`|`ml.g5.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.48xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 11.5 seconds`, for an `average prompt size of 304 tokens` and `completion size of 81 tokens` with `27 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`ml.g5.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.48xlarge` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 15.43 seconds`, for an `average prompt size of 1643 tokens` and `completion size of 84 tokens` with `6 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`ml.g5.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.48xlarge` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 12.21 seconds`, for an `average prompt size of 2503 tokens` and `completion size of 70 tokens` with `4 transactions/minute`.|
|`payload_en_3000-4000.jsonl`|`ml.g5.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.48xlarge` for the `payload_en_3000-4000.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 15.44 seconds`, for an `average prompt size of 3478 tokens` and `completion size of 67 tokens` with `3 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`ml.g5.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.48xlarge` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 4`. A concurrency level of 4 achieves an `average latency of 18.49 seconds`, for an `average prompt size of 980 tokens` and `completion size of 101 tokens` with `13 transactions/minute`.|
|`payload_en_1-500.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.p4d.24xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 1.93 seconds`, for an `average prompt size of 304 tokens` and `completion size of 328 tokens` with `131 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.p4d.24xlarge` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 5.28 seconds`, for an `average prompt size of 1630 tokens` and `completion size of 1660 tokens` with `50 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.p4d.24xlarge` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 7.66 seconds`, for an `average prompt size of 2503 tokens` and `completion size of 2533 tokens` with `34 transactions/minute`.|
|`payload_en_3000-4000.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.p4d.24xlarge` for the `payload_en_3000-4000.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 9.53 seconds`, for an `average prompt size of 3455 tokens` and `completion size of 3484 tokens` with `28 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.p4d.24xlarge` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 3.06 seconds`, for an `average prompt size of 980 tokens` and `completion size of 1004 tokens` with `93 transactions/minute`.|

## Plots

The following plots provide insights into the results from the different experiments run.

![Error rates for different concurrency levels and instance types](error_rates.png)

![Tokens vs latency for different concurrency levels and instance types](tokens_vs_latency.png)

![Concurrency Vs latency for different instance type for selected dataset](concurrency_vs_inference_latency.png)
