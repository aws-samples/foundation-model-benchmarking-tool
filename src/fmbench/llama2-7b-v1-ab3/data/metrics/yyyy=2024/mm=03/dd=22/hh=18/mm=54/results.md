
# Results for performance benchmarking

**Last modified (UTC): 2024-03-22 23:01:15.987741**

## Summary

We did performance benchmarking for the `Llama2-7b` model on "`ml.g5.xlarge`" instance on multiple datasets and based on the test results the best price performance for dataset `en_1000-2000` is provided by the `ml.g5.xlarge` instance type.  
| Information | Value |
|-----|-----|
| experiment_name | llama2-7b-g5.xlarge-huggingface-pytorch-tgi-inference-2.0.1-tgi1.1.0 |
| payload_file | payload_en_1000-2000.jsonl |
| instance_type | ml.g5.xlarge |
| concurrency | 1 |
| error_rate | 0.0 |
| prompt_token_count_mean | 1623 |
| prompt_token_throughput | 1265 |
| completion_token_count_mean | 53 |
| completion_token_throughput | 18 |
| latency_mean | 2.35 |
| transactions_per_minute | 49 |
| price_per_hour | 1.006 |
| price_per_txn | 0.000342 |


The price performance comparison for different instance types is presented below:

![Price performance comparison](business_summary.png)

The configuration used for these tests is available in the [`config`](config-llama2-7b-g5-quick.yml) file.

The cost to run each experiment is provided in the table below. The total cost for running all experiments is $0.01.

| experiment_name | instance_type | duration_in_seconds | cost |
|-----|-----|-----|-----|
| llama2-7b-g5.xlarge-huggingface-pytorch-tgi-inference-2.0.1-tgi1.1.0 | ml.g5.xlarge | 42.78 | 0.01 |




## Per instance results

The following table provides the best combinations for running inference for different sizes prompts on different instance types.

|Dataset   | Instance type   | Recommendation   |
|---|---|---|
|`payload_en_1000-2000.jsonl`|`ml.g5.xlarge`|The best option for staying within a latency budget of `20 seconds` on a `ml.g5.xlarge` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 2.35 seconds`, for an `average prompt size of 1623 tokens` and `completion size of 53 tokens` with `49 transactions/minute`.|

## Plots

The following plots provide insights into the results from the different experiments run.

![Error rates for different concurrency levels and instance types](error_rates.png)

![Tokens vs latency for different concurrency levels and instance types](tokens_vs_latency.png)

![Concurrency Vs latency for different instance type for selected dataset](concurrency_vs_inference_latency.png)
