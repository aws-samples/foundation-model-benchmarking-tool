
# Results for performance benchmarking

**Last modified (UTC): 2024-01-20 16:17:48.402896**

## Summary

We did performance benchmarking for the `Llama2-13b` model on "`ml.g5.12xlarge`, `ml.g5.24xlarge`, `ml.g5.48xlarge`, `ml.inf2.24xlarge`, `ml.inf2.48xlarge`, `ml.p4d.24xlarge`" instances on multiple datasets and based on the test results the best price performance for dataset `en_3000-4000` is provided by the `ml.inf2.48xlarge` instance type.  
| Information | Value |
|-----|-----|
| experiment_name | llama2-13b-inf2.48xlarge-djl-0.24.0-neuronx-sdk-2.14.1-bs=4-tpd=24 |
| payload_file | payload_en_3000-4000.jsonl |
| instance_type | ml.inf2.48xlarge |
| concurrency | 2 |
| error_rate | 0.0 |
| prompt_token_count_mean | 3482 |
| prompt_token_throughput | 3453 |
| completion_token_count_mean | 37 |
| completion_token_throughput | 33 |
| latency_mean | 1.85 |
| transactions_per_minute | 59 |
| price_per_hour | 15.58 |
| price_per_txn | 0.0044 |


The price performance comparison for different instance types is presented below:

![Price performance comparison](business_summary.png)

The configuration used for these tests is available in the [`config`](config-llama2-13b-inf2-g5-p4d-v1.yml) file.


## Per instance results

The following table provides the best combinations for running inference for different sizes prompts on different instance types.
|Dataset   | Instance type   | Recommendation   |
|---|---|---|
|`payload_en_1-500.jsonl`|`ml.g5.12xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.12xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 2.31 seconds`, for an `average prompt size of 304 tokens` and `completion size of 102 tokens` with `25 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`ml.g5.12xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.12xlarge` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 4`. A concurrency level of 4 achieves an `average latency of 4.78 seconds`, for an `average prompt size of 1548 tokens` and `completion size of 77 tokens` with `40 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`ml.g5.12xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.12xlarge` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 4`. A concurrency level of 4 achieves an `average latency of 7.05 seconds`, for an `average prompt size of 2503 tokens` and `completion size of 84 tokens` with `29 transactions/minute`.|
|`payload_en_3000-4000.jsonl`|`ml.g5.12xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.12xlarge` for the `payload_en_3000-4000.jsonl` dataset is a `concurrency level of 4`. A concurrency level of 4 achieves an `average latency of 8.94 seconds`, for an `average prompt size of 3482 tokens` and `completion size of 82 tokens` with `22 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`ml.g5.12xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.12xlarge` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 2.79 seconds`, for an `average prompt size of 980 tokens` and `completion size of 102 tokens` with `21 transactions/minute`.|
|`payload_en_1-500.jsonl`|`ml.g5.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.24xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 2.31 seconds`, for an `average prompt size of 304 tokens` and `completion size of 102 tokens` with `25 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`ml.g5.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.24xlarge` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 4`. A concurrency level of 4 achieves an `average latency of 4.81 seconds`, for an `average prompt size of 1548 tokens` and `completion size of 77 tokens` with `40 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`ml.g5.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.24xlarge` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 4`. A concurrency level of 4 achieves an `average latency of 7.15 seconds`, for an `average prompt size of 2503 tokens` and `completion size of 84 tokens` with `28 transactions/minute`.|
|`payload_en_3000-4000.jsonl`|`ml.g5.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.24xlarge` for the `payload_en_3000-4000.jsonl` dataset is a `concurrency level of 4`. A concurrency level of 4 achieves an `average latency of 9.48 seconds`, for an `average prompt size of 3482 tokens` and `completion size of 90 tokens` with `21 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`ml.g5.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.24xlarge` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 2.77 seconds`, for an `average prompt size of 980 tokens` and `completion size of 102 tokens` with `21 transactions/minute`.|
|`payload_en_1-500.jsonl`|`ml.g5.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.48xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 2.26 seconds`, for an `average prompt size of 304 tokens` and `completion size of 102 tokens` with `26 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`ml.g5.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.48xlarge` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 4`. A concurrency level of 4 achieves an `average latency of 5.85 seconds`, for an `average prompt size of 1548 tokens` and `completion size of 87 tokens` with `35 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`ml.g5.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.48xlarge` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 4`. A concurrency level of 4 achieves an `average latency of 8.87 seconds`, for an `average prompt size of 2503 tokens` and `completion size of 91 tokens` with `23 transactions/minute`.|
|`payload_en_3000-4000.jsonl`|`ml.g5.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.48xlarge` for the `payload_en_3000-4000.jsonl` dataset is a `concurrency level of 4`. A concurrency level of 4 achieves an `average latency of 12.26 seconds`, for an `average prompt size of 3482 tokens` and `completion size of 92 tokens` with `17 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`ml.g5.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.48xlarge` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 2.61 seconds`, for an `average prompt size of 980 tokens` and `completion size of 102 tokens` with `22 transactions/minute`.|
|`payload_en_1-500.jsonl`|`ml.inf2.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.inf2.24xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 2.58 seconds`, for an `average prompt size of 304 tokens` and `completion size of 24 tokens` with `23 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`ml.inf2.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.inf2.24xlarge` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 4`. A concurrency level of 4 achieves an `average latency of 5.33 seconds`, for an `average prompt size of 1548 tokens` and `completion size of 64 tokens` with `35 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`ml.inf2.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.inf2.24xlarge` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 4`. A concurrency level of 4 achieves an `average latency of 5.4 seconds`, for an `average prompt size of 2503 tokens` and `completion size of 64 tokens` with `36 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`ml.inf2.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.inf2.24xlarge` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 1.22 seconds`, for an `average prompt size of 980 tokens` and `completion size of 16 tokens` with `48 transactions/minute`.|
|`payload_en_1-500.jsonl`|`ml.inf2.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.inf2.48xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 3.62 seconds`, for an `average prompt size of 304 tokens` and `completion size of 25 tokens` with `16 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`ml.inf2.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.inf2.48xlarge` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 4`. A concurrency level of 4 achieves an `average latency of 3.81 seconds`, for an `average prompt size of 1548 tokens` and `completion size of 67 tokens` with `52 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`ml.inf2.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.inf2.48xlarge` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 4`. A concurrency level of 4 achieves an `average latency of 3.73 seconds`, for an `average prompt size of 2503 tokens` and `completion size of 63 tokens` with `52 transactions/minute`.|
|`payload_en_3000-4000.jsonl`|`ml.inf2.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.inf2.48xlarge` for the `payload_en_3000-4000.jsonl` dataset is a `concurrency level of 4`. A concurrency level of 4 achieves an `average latency of 3.58 seconds`, for an `average prompt size of 3482 tokens` and `completion size of 56 tokens` with `54 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`ml.inf2.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.inf2.48xlarge` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 0.85 seconds`, for an `average prompt size of 980 tokens` and `completion size of 16 tokens` with `70 transactions/minute`.|
|`payload_en_1-500.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.p4d.24xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 0.8 seconds`, for an `average prompt size of 304 tokens` and `completion size of 25 tokens` with `49 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.p4d.24xlarge` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 4`. A concurrency level of 4 achieves an `average latency of 1.5 seconds`, for an `average prompt size of 1548 tokens` and `completion size of 31 tokens` with `122 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.p4d.24xlarge` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 4`. A concurrency level of 4 achieves an `average latency of 2.3 seconds`, for an `average prompt size of 2503 tokens` and `completion size of 41 tokens` with `74 transactions/minute`.|
|`payload_en_3000-4000.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.p4d.24xlarge` for the `payload_en_3000-4000.jsonl` dataset is a `concurrency level of 4`. A concurrency level of 4 achieves an `average latency of 2.87 seconds`, for an `average prompt size of 3482 tokens` and `completion size of 33 tokens` with `58 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.p4d.24xlarge` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 0.41 seconds`, for an `average prompt size of 980 tokens` and `completion size of 15 tokens` with `143 transactions/minute`.|

## Plots

The following plots provide insights into the results from the different experiments run.

![Error rates for different concurrency levels and instance types](error_rates.png)

![Tokens vs latency for different concurrency levels and instance types](tokens_vs_latency.png)

![Concurrency Vs latency for different instance type for selected dataset](concurrency_vs_inference_latency.png)
