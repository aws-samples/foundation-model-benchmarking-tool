
# Results for performance benchmarking

**Last modified (UTC): 2024-04-16 14:14:05.831199**

## Summary

We did performance benchmarking for `Llama2-70b` on "`ml.g5.48xlarge`, `ml.p4d.24xlarge`" on multiple datasets and based on the test results the best price performance for dataset `en_3000-3840` is provided by the `ml.p4d.24xlarge`.


| Information | Value |
|-----|-----|
| experiment_name | llama2-70b-p4d.24xlarge-tgi-inference-2.0.1-tgi0.9.3-gpu-py39-cu118 |
| payload_file | payload_en_3000-3840.jsonl |
| instance_type | ml.p4d.24xlarge |
| concurrency | 1 |
| error_rate | 0.0 |
| prompt_token_count_mean | 3394 |
| prompt_token_throughput | 1473 |
| completion_token_count_mean | 3420 |
| completion_token_throughput | 1481 |
| latency_mean | 2.6 |
| latency_p50 | 2.6 |
| latency_p95 | 2.6 |
| latency_p99 | 2.6 |
| transactions_per_minute | 25 |
| price_per_txn | 0.025125 |


The price performance comparison for different instance types is presented below:

![Price performance comparison](business_summary.png)

There were a total of 10 experiment run(s) that failed at least one configured performance criteria: `Latency` < `5s`, `cost per 10k transactions`: `$50`, `error rate`: `0`. See table below.    
    

| experiment_name | payload_file | concurrency | error_rate_text | latency_mean_text | price_per_10k_txn_text |
|-----|-----|-----|-----|-----|-----|
| llama2-70b-g5.48xlarge-huggingface-pytorch-tgi-inference-2.0.1-tgi1.1.0 | payload_en_3000-3840.jsonl | 1 | <span style='color:green'>0.00</span> | <span style='color:red'>**10.71**</span> | <span style='color:red'>**678.67**</span> |
| llama2-70b-g5.48xlarge-huggingface-pytorch-tgi-inference-2.0.1-tgi1.1.0 | payload_en_3000-3840.jsonl | 2 | <span style='color:green'>0.00</span> | <span style='color:red'>**19.35**</span> | <span style='color:red'>**678.67**</span> |
| llama2-70b-g5.48xlarge-huggingface-pytorch-tgi-inference-2.0.1-tgi1.1.0 | payload_en_3000-3840.jsonl | 4 | <span style='color:green'>0.00</span> | <span style='color:red'>**31.97**</span> | <span style='color:red'>**678.67**</span> |
| llama2-70b-g5.48xlarge-huggingface-pytorch-tgi-inference-2.0.1-tgi1.1.0 | payload_en_3000-3840.jsonl | 6 | <span style='color:green'>0.00</span> | <span style='color:red'>**42.79**</span> | <span style='color:red'>**565.56**</span> |
| llama2-70b-g5.48xlarge-huggingface-pytorch-tgi-inference-2.0.1-tgi1.1.0 | payload_en_3000-3840.jsonl | 8 | <span style='color:green'>0.00</span> | <span style='color:red'>**55.85**</span> | <span style='color:red'>**678.67**</span> |
| llama2-70b-p4d.24xlarge-tgi-inference-2.0.1-tgi0.9.3-gpu-py39-cu118 | payload_en_3000-3840.jsonl | 1 | <span style='color:green'>0.00</span> | <span style='color:green'>2.60</span> | <span style='color:red'>**251.25**</span> |
| llama2-70b-p4d.24xlarge-tgi-inference-2.0.1-tgi0.9.3-gpu-py39-cu118 | payload_en_3000-3840.jsonl | 2 | <span style='color:green'>0.00</span> | <span style='color:green'>3.80</span> | <span style='color:red'>**209.38**</span> |
| llama2-70b-p4d.24xlarge-tgi-inference-2.0.1-tgi0.9.3-gpu-py39-cu118 | payload_en_3000-3840.jsonl | 4 | <span style='color:green'>0.00</span> | <span style='color:red'>**6.09**</span> | <span style='color:red'>**216.60**</span> |
| llama2-70b-p4d.24xlarge-tgi-inference-2.0.1-tgi0.9.3-gpu-py39-cu118 | payload_en_3000-3840.jsonl | 6 | <span style='color:green'>0.00</span> | <span style='color:red'>**8.09**</span> | <span style='color:red'>**216.60**</span> |
| llama2-70b-p4d.24xlarge-tgi-inference-2.0.1-tgi0.9.3-gpu-py39-cu118 | payload_en_3000-3840.jsonl | 8 | <span style='color:green'>0.00</span> | <span style='color:red'>**10.29**</span> | <span style='color:red'>**209.38**</span> |


The configuration used for these tests is available in the [`config`](config-llama2-70b-g5-p4d-tgi.yml) file.

The cost to run each experiment is provided in the table below. The total cost for running all experiments is $35.76.



| experiment_name | instance_type | duration_in_seconds | cost |
|-----|-----|-----|-----|
| llama2-70b-g5.48xlarge-huggingface-pytorch-tgi-inference-2.0.1-tgi1.1.0 | ml.g5.48xlarge | 4344.23 | 24.569008 |
| llama2-70b-p4d.24xlarge-tgi-inference-2.0.1-tgi0.9.3-gpu-py39-cu118 | ml.p4d.24xlarge | 1069.4 | 11.195401 |




## Per instance results

The following table provides the best combinations for running inference for different sizes prompts on different instance types. The following dataset(s) were used for this test: `2wikimqa_e.jsonl`, `2wikimqa.jsonl`, `hotpotqa_e.jsonl`, `hotpotqa.jsonl`, `narrativeqa.jsonl`, `triviaqa_e.jsonl`, `triviaqa.jsonl`.

|Dataset   | Instance type   | Recommendation   |
|---|---|---|
|`payload_en_1-500.jsonl`|`ml.g5.48xlarge`|The best option for staying within a latency budget of `5 seconds` on a `ml.g5.48xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 4`. A concurrency level of 4 achieves an `average latency of 4.25 seconds`, for an `average prompt size of 304 tokens` and `completion size of 26 tokens` with `56 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`ml.g5.48xlarge`|This experiment did not find any combination of concurrency level and other configuration settings that could provide a response within a latency budget of `5 seconds` on a `ml.g5.48xlarge` for the `payload_en_1000-2000.jsonl` dataset.|
|`payload_en_2000-3000.jsonl`|`ml.g5.48xlarge`|This experiment did not find any combination of concurrency level and other configuration settings that could provide a response within a latency budget of `5 seconds` on a `ml.g5.48xlarge` for the `payload_en_2000-3000.jsonl` dataset.|
|`payload_en_3000-3840.jsonl`|`ml.g5.48xlarge`|This experiment did not find any combination of concurrency level and other configuration settings that could provide a response within a latency budget of `5 seconds` on a `ml.g5.48xlarge` for the `payload_en_3000-3840.jsonl` dataset.|
|`payload_en_500-1000.jsonl`|`ml.g5.48xlarge`|The best option for staying within a latency budget of `5 seconds` on a `ml.g5.48xlarge` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 3.56 seconds`, for an `average prompt size of 980 tokens` and `completion size of 25 tokens` with `16 transactions/minute`.|
|`payload_en_1-500.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `5 seconds` on a `ml.p4d.24xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 2.27 seconds`, for an `average prompt size of 304 tokens` and `completion size of 328 tokens` with `156 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `5 seconds` on a `ml.p4d.24xlarge` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 6`. A concurrency level of 6 achieves an `average latency of 4.24 seconds`, for an `average prompt size of 1672 tokens` and `completion size of 1696 tokens` with `58 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `5 seconds` on a `ml.p4d.24xlarge` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 3.44 seconds`, for an `average prompt size of 2503 tokens` and `completion size of 2539 tokens` with `32 transactions/minute`.|
|`payload_en_3000-3840.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `5 seconds` on a `ml.p4d.24xlarge` for the `payload_en_3000-3840.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 3.8 seconds`, for an `average prompt size of 3394 tokens` and `completion size of 3419 tokens` with `30 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `5 seconds` on a `ml.p4d.24xlarge` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 8`. A concurrency level of 8 achieves an `average latency of 4.07 seconds`, for an `average prompt size of 980 tokens` and `completion size of 1004 tokens` with `85 transactions/minute`.|

## Plots

The following plots provide insights into the results from the different experiments run.

![Error rates for different concurrency levels and instance types](error_rates.png)

![Tokens vs latency for different concurrency levels and instance types](tokens_vs_latency.png)

![Concurrency Vs latency for different instance type for selected dataset](concurrency_vs_inference_latency.png)
