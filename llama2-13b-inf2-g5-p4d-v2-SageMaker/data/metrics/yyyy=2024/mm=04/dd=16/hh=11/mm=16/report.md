
# Results for performance benchmarking

**Last modified (UTC): 2024-04-16 11:49:01.507629**

## Summary

We did performance benchmarking for `Llama2-13b` on "`ml.g5.12xlarge`, `ml.g5.24xlarge`, `ml.g5.48xlarge`, `ml.inf2.48xlarge`, `ml.p4d.24xlarge`" on multiple datasets and based on the test results the best price performance for dataset `en_3000-3840` is provided by the `ml.inf2.48xlarge`.


| Information | Value |
|-----|-----|
| experiment_name | llama2-13b-inf2.48xlarge-djl-0.24.0-neuronx-sdk-2.14.1-bs=4-tpd=24 |
| payload_file | payload_en_3000-3840.jsonl |
| instance_type | ml.inf2.48xlarge |
| concurrency | 2 |
| error_rate | 0.0 |
| prompt_token_count_mean | 3394 |
| prompt_token_throughput | 3805 |
| completion_token_count_mean | 30 |
| completion_token_throughput | 29 |
| latency_mean | 1.7 |
| latency_p50 | 1.7 |
| latency_p95 | 1.88 |
| latency_p99 | 1.9 |
| transactions_per_minute | 66 |
| price_per_txn | 0.003934 |


The price performance comparison for different instance types is presented below:

![Price performance comparison](business_summary.png)

There were a total of 7 experiment run(s) that failed at least one configured performance criteria: `Latency` < `2s`, `cost per 10k transactions`: `$50`, `error rate`: `0`. See table below.    
    

| experiment_name | payload_file | concurrency | error_rate_text | latency_mean_text | price_per_10k_txn_text |
|-----|-----|-----|-----|-----|-----|
| llama2-13b-g5.12xlarge-huggingface-pytorch-tgi-inference-2.0.1-tgi1.1.0 | payload_en_3000-3840.jsonl | 2 | <span style='color:green'>0.00</span> | <span style='color:red'>**3.37**</span> | <span style='color:green'>35.81</span> |
| llama2-13b-g5.24xlarge-huggingface-pytorch-tgi-inference-2.0.1-tgi1.1.0 | payload_en_3000-3840.jsonl | 1 | <span style='color:green'>0.00</span> | <span style='color:red'>**2.01**</span> | <span style='color:red'>**54.73**</span> |
| llama2-13b-g5.24xlarge-huggingface-pytorch-tgi-inference-2.0.1-tgi1.1.0 | payload_en_3000-3840.jsonl | 2 | <span style='color:green'>0.00</span> | <span style='color:red'>**3.42**</span> | <span style='color:red'>**51.41**</span> |
| llama2-13b-g5.48xlarge-huggingface-pytorch-tgi-inference-2.0.1-tgi1.1.0 | payload_en_3000-3840.jsonl | 1 | <span style='color:green'>0.00</span> | <span style='color:red'>**3.04**</span> | <span style='color:red'>**178.60**</span> |
| llama2-13b-g5.48xlarge-huggingface-pytorch-tgi-inference-2.0.1-tgi1.1.0 | payload_en_3000-3840.jsonl | 2 | <span style='color:green'>0.00</span> | <span style='color:red'>**5.48**</span> | <span style='color:red'>**169.67**</span> |
| llama2-13b-p4d.24xlarge-tgi-inference-2.0.1-tgi0.9.3-gpu-py39-cu118 | payload_en_3000-3840.jsonl | 1 | <span style='color:green'>0.00</span> | <span style='color:green'>0.96</span> | <span style='color:red'>**74.78**</span> |
| llama2-13b-p4d.24xlarge-tgi-inference-2.0.1-tgi0.9.3-gpu-py39-cu118 | payload_en_3000-3840.jsonl | 2 | <span style='color:green'>0.00</span> | <span style='color:green'>1.23</span> | <span style='color:red'>**62.81**</span> |


The configuration used for these tests is available in the [`config`](config-llama2-13b-inf2-g5-p4d-quick.yml) file.

The cost to run each experiment is provided in the table below. The total cost for running all experiments is $8.21.



| experiment_name | instance_type | duration_in_seconds | cost |
|-----|-----|-----|-----|
| llama2-13b-inf2.48xlarge-djl-0.24.0-neuronx-sdk-2.14.1-bs=4-tpd=24 | ml.inf2.48xlarge | 279.75 | 1.210717 |
| llama2-13b-p4d.24xlarge-tgi-inference-2.0.1-tgi0.9.3-gpu-py39-cu118 | ml.p4d.24xlarge | 206.29 | 2.159653 |
| llama2-13b-g5.12xlarge-huggingface-pytorch-tgi-inference-2.0.1-tgi1.1.0 | ml.g5.12xlarge | 370.36 | 0.729396 |
| llama2-13b-g5.24xlarge-huggingface-pytorch-tgi-inference-2.0.1-tgi1.1.0 | ml.g5.24xlarge | 377.85 | 1.068479 |
| llama2-13b-g5.48xlarge-huggingface-pytorch-tgi-inference-2.0.1-tgi1.1.0 | ml.g5.48xlarge | 537.07 | 3.037439 |




## Per instance results

The following table provides the best combinations for running inference for different sizes prompts on different instance types. The following dataset(s) were used for this test: `2wikimqa_e.jsonl`, `2wikimqa.jsonl`, `hotpotqa_e.jsonl`, `hotpotqa.jsonl`, `narrativeqa.jsonl`, `triviaqa_e.jsonl`, `triviaqa.jsonl`.

|Dataset   | Instance type   | Recommendation   |
|---|---|---|
|`payload_en_1-500.jsonl`|`ml.g5.12xlarge`|The best option for staying within a latency budget of `2 seconds` on a `ml.g5.12xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 0.8 seconds`, for an `average prompt size of 304 tokens` and `completion size of 26 tokens` with `149 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`ml.g5.12xlarge`|The best option for staying within a latency budget of `2 seconds` on a `ml.g5.12xlarge` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 1.32 seconds`, for an `average prompt size of 1623 tokens` and `completion size of 32 tokens` with `49 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`ml.g5.12xlarge`|The best option for staying within a latency budget of `2 seconds` on a `ml.g5.12xlarge` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 1.92 seconds`, for an `average prompt size of 2503 tokens` and `completion size of 44 tokens` with `34 transactions/minute`.|
|`payload_en_3000-3840.jsonl`|`ml.g5.12xlarge`|The best option for staying within a latency budget of `2 seconds` on a `ml.g5.12xlarge` for the `payload_en_3000-3840.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 1.99 seconds`, for an `average prompt size of 3394 tokens` and `completion size of 30 tokens` with `31 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`ml.g5.12xlarge`|The best option for staying within a latency budget of `2 seconds` on a `ml.g5.12xlarge` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.09 seconds`, for an `average prompt size of 980 tokens` and `completion size of 16 tokens` with `109 transactions/minute`.|
|`payload_en_1-500.jsonl`|`ml.g5.24xlarge`|The best option for staying within a latency budget of `2 seconds` on a `ml.g5.24xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 0.78 seconds`, for an `average prompt size of 304 tokens` and `completion size of 25 tokens` with `150 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`ml.g5.24xlarge`|The best option for staying within a latency budget of `2 seconds` on a `ml.g5.24xlarge` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 1.36 seconds`, for an `average prompt size of 1623 tokens` and `completion size of 34 tokens` with `47 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`ml.g5.24xlarge`|The best option for staying within a latency budget of `2 seconds` on a `ml.g5.24xlarge` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 1.92 seconds`, for an `average prompt size of 2503 tokens` and `completion size of 43 tokens` with `34 transactions/minute`.|
|`payload_en_3000-3840.jsonl`|`ml.g5.24xlarge`|This experiment did not find any combination of concurrency level and other configuration settings that could provide a response within a latency budget of `2 seconds` on a `ml.g5.24xlarge` for the `payload_en_3000-3840.jsonl` dataset.|
|`payload_en_500-1000.jsonl`|`ml.g5.24xlarge`|The best option for staying within a latency budget of `2 seconds` on a `ml.g5.24xlarge` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.1 seconds`, for an `average prompt size of 980 tokens` and `completion size of 16 tokens` with `108 transactions/minute`.|
|`payload_en_1-500.jsonl`|`ml.g5.48xlarge`|The best option for staying within a latency budget of `2 seconds` on a `ml.g5.48xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 0.94 seconds`, for an `average prompt size of 304 tokens` and `completion size of 25 tokens` with `124 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`ml.g5.48xlarge`|The best option for staying within a latency budget of `2 seconds` on a `ml.g5.48xlarge` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 1.86 seconds`, for an `average prompt size of 1623 tokens` and `completion size of 35 tokens` with `34 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`ml.g5.48xlarge`|This experiment did not find any combination of concurrency level and other configuration settings that could provide a response within a latency budget of `2 seconds` on a `ml.g5.48xlarge` for the `payload_en_2000-3000.jsonl` dataset.|
|`payload_en_3000-3840.jsonl`|`ml.g5.48xlarge`|This experiment did not find any combination of concurrency level and other configuration settings that could provide a response within a latency budget of `2 seconds` on a `ml.g5.48xlarge` for the `payload_en_3000-3840.jsonl` dataset.|
|`payload_en_500-1000.jsonl`|`ml.g5.48xlarge`|The best option for staying within a latency budget of `2 seconds` on a `ml.g5.48xlarge` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.64 seconds`, for an `average prompt size of 980 tokens` and `completion size of 16 tokens` with `72 transactions/minute`.|
|`payload_en_1-500.jsonl`|`ml.inf2.48xlarge`|The best option for staying within a latency budget of `2 seconds` on a `ml.inf2.48xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.67 seconds`, for an `average prompt size of 304 tokens` and `completion size of 26 tokens` with `71 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`ml.inf2.48xlarge`|The best option for staying within a latency budget of `2 seconds` on a `ml.inf2.48xlarge` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.85 seconds`, for an `average prompt size of 1643 tokens` and `completion size of 36 tokens` with `61 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`ml.inf2.48xlarge`|The best option for staying within a latency budget of `2 seconds` on a `ml.inf2.48xlarge` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 1.38 seconds`, for an `average prompt size of 2503 tokens` and `completion size of 46 tokens` with `49 transactions/minute`.|
|`payload_en_3000-3840.jsonl`|`ml.inf2.48xlarge`|The best option for staying within a latency budget of `2 seconds` on a `ml.inf2.48xlarge` for the `payload_en_3000-3840.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.7 seconds`, for an `average prompt size of 3394 tokens` and `completion size of 30 tokens` with `66 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`ml.inf2.48xlarge`|The best option for staying within a latency budget of `2 seconds` on a `ml.inf2.48xlarge` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.47 seconds`, for an `average prompt size of 980 tokens` and `completion size of 16 tokens` with `81 transactions/minute`.|
|`payload_en_1-500.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `2 seconds` on a `ml.p4d.24xlarge` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 0.76 seconds`, for an `average prompt size of 304 tokens` and `completion size of 26 tokens` with `150 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `2 seconds` on a `ml.p4d.24xlarge` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 0.99 seconds`, for an `average prompt size of 1643 tokens` and `completion size of 35 tokens` with `114 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `2 seconds` on a `ml.p4d.24xlarge` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.29 seconds`, for an `average prompt size of 2503 tokens` and `completion size of 46 tokens` with `84 transactions/minute`.|
|`payload_en_3000-3840.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `2 seconds` on a `ml.p4d.24xlarge` for the `payload_en_3000-3840.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.23 seconds`, for an `average prompt size of 3394 tokens` and `completion size of 30 tokens` with `100 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`ml.p4d.24xlarge`|The best option for staying within a latency budget of `2 seconds` on a `ml.p4d.24xlarge` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 0.49 seconds`, for an `average prompt size of 980 tokens` and `completion size of 16 tokens` with `243 transactions/minute`.|

## Plots

The following plots provide insights into the results from the different experiments run.

![Error rates for different concurrency levels and instance types](error_rates.png)

![Tokens vs latency for different concurrency levels and instance types](tokens_vs_latency.png)

![Concurrency Vs latency for different instance type for selected dataset](concurrency_vs_inference_latency.png)
