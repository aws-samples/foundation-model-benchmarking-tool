
# Results for performance benchmarking

**Last modified (UTC): 2024-01-21 05:30:11.627159**

## Summary

We did performance benchmarking for the `Llama2-13b` model on "`ml.g5.12xlarge`, `ml.g5.24xlarge`, `ml.g5.48xlarge`, `ml.inf2.24xlarge`, `ml.inf2.48xlarge`, `ml.p4d.24xlarge`" instances on multiple datasets and based on the test results the best price performance for dataset `en_3000-4000` is provided by the `ml.inf2.48xlarge` instance type.  
| Information | Value |
|-----|-----|
| experiment_name | llama2-13b-inf2.48xlarge-djl-0.24.0-neuronx-sdk-2.14.1-bs=4-tpd=24 |
| payload_file | payload_en_3000-4000.jsonl |
| instance_type | ml.inf2.48xlarge |
| concurrency | 8 |
| error_rate | 0.0 |
| prompt_token_count_mean | 3455 |
| prompt_token_throughput | 3867 |
| completion_token_count_mean | 43 |
| completion_token_throughput | 46 |
| latency_mean | 4.51 |
| transactions_per_minute | 66 |
| price_per_hour | 15.58 |
| price_per_txn | 0.0039 |


The price performance comparison for different instance types is presented below:

![Price performance comparison](business_summary.png)

The configuration used for these tests is available in the [`config`](config-llama2-13b-inf2-g5-p4d-v1.yml) file.


## Per instance results

The following table provides the best combinations for running inference for different sizes prompts on different instance types.
|Dataset   | Instance type   | Recommendation   |
|---|---|---|
|`payload_en_1-500.jsonl`|`ml.g5.12xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.12xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 2.6 seconds`, for an `average prompt size of 304 tokens` and `completion size of 78 tokens` with `123 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`ml.g5.12xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.12xlarge` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 6.67 seconds`, for an `average prompt size of 1630 tokens` and `completion size of 84 tokens` with `43 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`ml.g5.12xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.12xlarge` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 9.61 seconds`, for an `average prompt size of 2503 tokens` and `completion size of 85 tokens` with `31 transactions/minute`.|
|`payload_en_3000-4000.jsonl`|`ml.g5.12xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.12xlarge` for the `payload_en_3000-4000.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 12.9 seconds`, for an `average prompt size of 3455 tokens` and `completion size of 85 tokens` with `24 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`ml.g5.12xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.12xlarge` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 4.57 seconds`, for an `average prompt size of 980 tokens` and `completion size of 90 tokens` with `57 transactions/minute`.|
|`payload_en_1-500.jsonl`|`ml.g5.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.24xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 2.98 seconds`, for an `average prompt size of 304 tokens` and `completion size of 102 tokens` with `84 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`ml.g5.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.24xlarge` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 6.63 seconds`, for an `average prompt size of 1630 tokens` and `completion size of 80 tokens` with `47 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`ml.g5.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.24xlarge` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 9.66 seconds`, for an `average prompt size of 2503 tokens` and `completion size of 81 tokens` with `30 transactions/minute`.|
|`payload_en_3000-4000.jsonl`|`ml.g5.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.24xlarge` for the `payload_en_3000-4000.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 12.91 seconds`, for an `average prompt size of 3455 tokens` and `completion size of 82 tokens` with `24 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`ml.g5.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.24xlarge` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 4.72 seconds`, for an `average prompt size of 980 tokens` and `completion size of 84 tokens` with `66 transactions/minute`.|
|`payload_en_1-500.jsonl`|`ml.g5.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.48xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 3.19 seconds`, for an `average prompt size of 304 tokens` and `completion size of 90 tokens` with `79 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`ml.g5.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.48xlarge` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 9.54 seconds`, for an `average prompt size of 1630 tokens` and `completion size of 61 tokens` with `33 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`ml.g5.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.48xlarge` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 15.72 seconds`, for an `average prompt size of 2503 tokens` and `completion size of 85 tokens` with `20 transactions/minute`.|
|`payload_en_3000-4000.jsonl`|`ml.g5.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.48xlarge` for the `payload_en_3000-4000.jsonl` dataset is a `concurrency level of 4`. A concurrency level of 4 achieves an `average latency of 14.41 seconds`, for an `average prompt size of 3468 tokens` and `completion size of 83 tokens` with `14 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`ml.g5.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.48xlarge` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 6.33 seconds`, for an `average prompt size of 980 tokens` and `completion size of 80 tokens` with `49 transactions/minute`.|
|`payload_en_1-500.jsonl`|`ml.inf2.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.inf2.24xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 5.02 seconds`, for an `average prompt size of 304 tokens` and `completion size of 25 tokens` with `64 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`ml.inf2.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.inf2.24xlarge` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 7.17 seconds`, for an `average prompt size of 1630 tokens` and `completion size of 57 tokens` with `44 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`ml.inf2.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.inf2.24xlarge` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 6.82 seconds`, for an `average prompt size of 2503 tokens` and `completion size of 54 tokens` with `43 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`ml.inf2.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.inf2.24xlarge` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 4.21 seconds`, for an `average prompt size of 980 tokens` and `completion size of 16 tokens` with `70 transactions/minute`.|
|`payload_en_1-500.jsonl`|`ml.inf2.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.inf2.48xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 3.84 seconds`, for an `average prompt size of 304 tokens` and `completion size of 25 tokens` with `84 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`ml.inf2.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.inf2.48xlarge` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 4.92 seconds`, for an `average prompt size of 1630 tokens` and `completion size of 52 tokens` with `63 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`ml.inf2.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.inf2.48xlarge` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 4.73 seconds`, for an `average prompt size of 2503 tokens` and `completion size of 49 tokens` with `62 transactions/minute`.|
|`payload_en_3000-4000.jsonl`|`ml.inf2.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.inf2.48xlarge` for the `payload_en_3000-4000.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 4.51 seconds`, for an `average prompt size of 3455 tokens` and `completion size of 43 tokens` with `66 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`ml.inf2.48xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.inf2.48xlarge` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 3.28 seconds`, for an `average prompt size of 980 tokens` and `completion size of 16 tokens` with `90 transactions/minute`.|
|`payload_en_1-500.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.p4d.24xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 0.82 seconds`, for an `average prompt size of 304 tokens` and `completion size of 327 tokens` with `325 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.p4d.24xlarge` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 2.16 seconds`, for an `average prompt size of 1630 tokens` and `completion size of 1665 tokens` with `126 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.p4d.24xlarge` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 3.34 seconds`, for an `average prompt size of 2503 tokens` and `completion size of 2542 tokens` with `82 transactions/minute`.|
|`payload_en_3000-4000.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.p4d.24xlarge` for the `payload_en_3000-4000.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 3.87 seconds`, for an `average prompt size of 3455 tokens` and `completion size of 3485 tokens` with `67 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.p4d.24xlarge` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 0.83 seconds`, for an `average prompt size of 980 tokens` and `completion size of 995 tokens` with `325 transactions/minute`.|

## Plots

The following plots provide insights into the results from the different experiments run.

![Error rates for different concurrency levels and instance types](error_rates.png)

![Tokens vs latency for different concurrency levels and instance types](tokens_vs_latency.png)

![Concurrency Vs latency for different instance type for selected dataset](concurrency_vs_inference_latency.png)
