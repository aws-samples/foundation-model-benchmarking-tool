
# Results for performance benchmarking

**Last modified (UTC): 2024-04-17 15:18:51.108681**

## Summary

We did performance benchmarking for `mistral7bInstruct-AWQ` on "`ml.p4d.24xlarge`" on multiple datasets and based on the test results the best price performance for dataset `en_1000-2000` is provided by the `ml.p4d.24xlarge`.


| Information | Value |
|-----|-----|
| experiment_name | mistral-Instruct7b-p4d-lmi-customer-drop |
| payload_file | payload_en_1000-2000.jsonl |
| instance_type | ml.p4d.24xlarge |
| concurrency | 30 |
| error_rate | 0.0 |
| prompt_token_count_mean | 1476 |
| prompt_token_throughput | 14552 |
| completion_token_count_mean | 32 |
| completion_token_throughput | 323 |
| latency_mean | 0.76 |
| latency_p50 | 0.64 |
| latency_p95 | 1.54 |
| latency_p99 | 1.67 |
| transactions_per_minute | 591 |
| price_per_txn | 0.001063 |


The price performance comparison for different instance types is presented below:

![Price performance comparison](business_summary.png)

There were a total of 3 experiment run(s) that failed at least one configured performance criteria: `Latency` < `1s`, `cost per 10k transactions`: `$20`, `error rate`: `0`. See table below.    
    

| experiment_name | payload_file | concurrency | error_rate_text | latency_mean_text | price_per_10k_txn_text |
|-----|-----|-----|-----|-----|-----|
| mistral-Instruct7b-p4d-lmi-customer-drop | payload_en_1000-2000.jsonl | 1 | <span style='color:green'>0.00</span> | <span style='color:green'>0.76</span> | <span style='color:red'>**76.60**</span> |
| mistral-Instruct7b-p4d-lmi-customer-drop | payload_en_1000-2000.jsonl | 2 | <span style='color:green'>0.00</span> | <span style='color:green'>0.77</span> | <span style='color:red'>**42.73**</span> |
| mistral-Instruct7b-p4d-lmi-customer-drop | payload_en_1000-2000.jsonl | 5 | <span style='color:green'>0.00</span> | <span style='color:green'>0.79</span> | <span style='color:red'>**22.12**</span> |


The configuration used for these tests is available in the [`config`](config-mistral-instruct-AWQ-p4d.yml) file.

The cost to run each experiment is provided in the table below. The total cost for running all experiments is $0.47.



| experiment_name | instance_type | duration_in_seconds | cost |
|-----|-----|-----|-----|
| mistral-Instruct7b-p4d-lmi-customer-drop | ml.p4d.24xlarge | 45.09 | 0.471999 |




## Per instance results

The following table provides the best combinations for running inference for different sizes prompts on different instance types. The following dataset(s) were used for this test: `2wikimqa_e.jsonl`, `2wikimqa.jsonl`, `hotpotqa_e.jsonl`, `hotpotqa.jsonl`, `narrativeqa.jsonl`, `triviaqa_e.jsonl`, `triviaqa.jsonl`.

|Dataset   | Instance type   | Recommendation   |
|---|---|---|
|`payload_en_1000-2000.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `1 seconds` on a `ml.p4d.24xlarge` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 30`. A concurrency level of 30 achieves an `average latency of 0.76 seconds`, for an `average prompt size of 1476 tokens` and `completion size of 32 tokens` with `591 transactions/minute`.|

## Plots

The following plots provide insights into the results from the different experiments run.

![Error rates for different concurrency levels and instance types](error_rates.png)

![Tokens vs latency for different concurrency levels and instance types](tokens_vs_latency.png)

![Concurrency Vs latency for different instance type for selected dataset](concurrency_vs_inference_latency.png)
