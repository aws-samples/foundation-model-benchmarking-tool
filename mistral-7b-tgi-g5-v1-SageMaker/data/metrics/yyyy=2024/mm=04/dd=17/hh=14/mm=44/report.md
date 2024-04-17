
# Results for performance benchmarking

**Last modified (UTC): 2024-04-17 14:49:45.408704**

## Summary

We did performance benchmarking for `mistral7b` on "`ml.g5.2xlarge`" on multiple datasets and based on the test results the best price performance for dataset `en_2000-3000` is provided by the `ml.g5.2xlarge`.


| Information | Value |
|-----|-----|
| experiment_name | mistral-7b--instruct-g5-huggingface-pytorch-tgi-inference-2.0.1-tgi1.1.0 |
| payload_file | payload_en_2000-3000.jsonl |
| instance_type | ml.g5.2xlarge |
| concurrency | 4 |
| error_rate | 0.0 |
| prompt_token_count_mean | 2517 |
| prompt_token_throughput | 2072 |
| completion_token_count_mean | 40 |
| completion_token_throughput | 32 |
| latency_mean | 3.89 |
| latency_p50 | 3.9 |
| latency_p95 | 4.85 |
| latency_p99 | 4.92 |
| transactions_per_minute | 48 |
| price_per_txn | 0.000421 |


The price performance comparison for different instance types is presented below:

![Price performance comparison](business_summary.png)

There were a total of 3 experiment run(s) that failed at least one configured performance criteria: `Latency` < `1s`, `cost per 10k transactions`: `$20`, `error rate`: `0`. See table below.    
    

| experiment_name | payload_file | concurrency | error_rate_text | latency_mean_text | price_per_10k_txn_text |
|-----|-----|-----|-----|-----|-----|
| mistral-7b--instruct-g5-huggingface-pytorch-tgi-inference-2.0.1-tgi1.1.0 | payload_en_2000-3000.jsonl | 1 | <span style='color:green'>0.00</span> | <span style='color:red'>**1.90**</span> | <span style='color:green'>5.05</span> |
| mistral-7b--instruct-g5-huggingface-pytorch-tgi-inference-2.0.1-tgi1.1.0 | payload_en_2000-3000.jsonl | 2 | <span style='color:green'>0.00</span> | <span style='color:red'>**2.50**</span> | <span style='color:green'>4.93</span> |
| mistral-7b--instruct-g5-huggingface-pytorch-tgi-inference-2.0.1-tgi1.1.0 | payload_en_2000-3000.jsonl | 4 | <span style='color:green'>0.00</span> | <span style='color:red'>**3.89**</span> | <span style='color:green'>4.21</span> |


The configuration used for these tests is available in the [`config`](config-mistral-7b-tgi-g5.yml) file.

The cost to run each experiment is provided in the table below. The total cost for running all experiments is $0.09.



| experiment_name | instance_type | duration_in_seconds | cost |
|-----|-----|-----|-----|
| mistral-7b--instruct-g5-huggingface-pytorch-tgi-inference-2.0.1-tgi1.1.0 | ml.g5.2xlarge | 262.97 | 0.088532 |




## Per instance results

The following table provides the best combinations for running inference for different sizes prompts on different instance types. The following dataset(s) were used for this test: `2wikimqa_e.jsonl`, `2wikimqa.jsonl`, `hotpotqa_e.jsonl`, `hotpotqa.jsonl`, `narrativeqa.jsonl`, `triviaqa_e.jsonl`, `triviaqa.jsonl`.

|Dataset   | Instance type   | Recommendation   |
|---|---|---|
|`payload_en_1-500.jsonl`|`ml.g5.2xlarge`|This experiment did not find any combination of concurrency level and other configuration settings that could provide a response within a latency budget of `1 seconds` on a `ml.g5.2xlarge` for the `payload_en_1-500.jsonl` dataset.|
|`payload_en_1000-2000.jsonl`|`ml.g5.2xlarge`|This experiment did not find any combination of concurrency level and other configuration settings that could provide a response within a latency budget of `1 seconds` on a `ml.g5.2xlarge` for the `payload_en_1000-2000.jsonl` dataset.|
|`payload_en_2000-3000.jsonl`|`ml.g5.2xlarge`|This experiment did not find any combination of concurrency level and other configuration settings that could provide a response within a latency budget of `1 seconds` on a `ml.g5.2xlarge` for the `payload_en_2000-3000.jsonl` dataset.|
|`payload_en_500-1000.jsonl`|`ml.g5.2xlarge`|This experiment did not find any combination of concurrency level and other configuration settings that could provide a response within a latency budget of `1 seconds` on a `ml.g5.2xlarge` for the `payload_en_500-1000.jsonl` dataset.|

## Plots

The following plots provide insights into the results from the different experiments run.

![Error rates for different concurrency levels and instance types](error_rates.png)

![Tokens vs latency for different concurrency levels and instance types](tokens_vs_latency.png)

![Concurrency Vs latency for different instance type for selected dataset](concurrency_vs_inference_latency.png)
