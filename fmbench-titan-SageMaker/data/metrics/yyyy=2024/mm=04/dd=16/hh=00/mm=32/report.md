
# Results for performance benchmarking

**Last modified (UTC): 2024-04-16 01:38:12.520678**

## Summary

We did performance benchmarking for `FMs available in Amazon Bedrock` on "`ai21.j2-mid-v1`, `ai21.j2-ultra-v1`, `amazon.titan-text-express-v1`, `amazon.titan-text-lite-v1`, `anthropic.claude-3-haiku-20240307-v1:0`, `anthropic.claude-3-sonnet-20240229-v1:0`, `cohere.command-light-text-v14`, `cohere.command-text-v14`, `meta.llama2-13b-chat-v1`, `mistral.mistral-7b-instruct-v0:2`, `mistral.mixtral-8x7b-instruct-v0:1`" on multiple datasets and based on the test results the best price performance for dataset `en_3000-4000` is provided by the `mistral.mistral-7b-instruct-v0:2`.


| Information | Value |
|-----|-----|
| experiment_name | mistral.mistral-7b-instruct-v0:2 |
| payload_file | payload_en_3000-4000.jsonl |
| instance_type | mistral.mistral-7b-instruct-v0:2 |
| concurrency | 2 |
| error_rate | 0.0 |
| prompt_token_count_mean | 2959 |
| prompt_token_throughput | 7326 |
| completion_token_count_mean | 34 |
| completion_token_throughput | 73 |
| latency_mean | 0.75 |
| latency_p50 | 0.75 |
| latency_p95 | 0.95 |
| latency_p99 | 0.97 |
| transactions_per_minute | 148 |
| price_per_txn | 0.000451 |


The price performance comparison for different instance types is presented below:

![Price performance comparison](business_summary.png)

There were a total of 10 experiment run(s) that failed at least one configured performance criteria: `Latency` < `2s`, `cost per 10k transactions`: `$50`, `error rate`: `0`. See table below.    
    

| experiment_name | payload_file | concurrency | error_rate_text | latency_mean_text | price_per_10k_txn_text |
|-----|-----|-----|-----|-----|-----|
| ai21.j2-mid-v1 | payload_en_3000-4000.jsonl | 1 | <span style='color:green'>0.00</span> | <span style='color:green'>0.76</span> | <span style='color:red'>**372.12**</span> |
| ai21.j2-mid-v1 | payload_en_3000-4000.jsonl | 2 | <span style='color:green'>0.00</span> | <span style='color:green'>0.73</span> | <span style='color:red'>**371.50**</span> |
| ai21.j2-ultra-v1 | payload_en_3000-4000.jsonl | 1 | <span style='color:green'>0.00</span> | <span style='color:green'>0.97</span> | <span style='color:red'>**558.92**</span> |
| ai21.j2-ultra-v1 | payload_en_3000-4000.jsonl | 2 | <span style='color:green'>0.00</span> | <span style='color:green'>0.98</span> | <span style='color:red'>**558.17**</span> |
| amazon.titan-text-lite-v1 | payload_en_3000-4000.jsonl | 1 | <span style='color:green'>0.00</span> | <span style='color:red'>**2.47**</span> | <span style='color:green'>8.92</span> |
| amazon.titan-text-lite-v1 | payload_en_3000-4000.jsonl | 2 | <span style='color:green'>0.00</span> | <span style='color:red'>**2.53**</span> | <span style='color:green'>8.91</span> |
| anthropic.claude-3-sonnet-20240229-v1:0 | payload_en_3000-4000.jsonl | 1 | <span style='color:green'>0.00</span> | <span style='color:red'>**4.85**</span> | <span style='color:red'>**109.02**</span> |
| anthropic.claude-3-sonnet-20240229-v1:0 | payload_en_3000-4000.jsonl | 2 | <span style='color:green'>0.00</span> | <span style='color:red'>**5.04**</span> | <span style='color:red'>**109.23**</span> |
| cohere.command-light-text-v14 | payload_en_3000-4000.jsonl | 1 | <span style='color:green'>0.00</span> | <span style='color:red'>**2.10**</span> | <span style='color:green'>9.13</span> |
| cohere.command-light-text-v14 | payload_en_3000-4000.jsonl | 2 | <span style='color:green'>0.00</span> | <span style='color:red'>**2.77**</span> | <span style='color:green'>9.13</span> |


The configuration used for these tests is available in the [`config`](config-bedrock.yml) file.

The cost to run each experiment is provided in the table below. The total cost for running all experiments is $21.17.



| experiment_name | instance_type | duration_in_seconds | cost |
|-----|-----|-----|-----|
| mistral.mistral-7b-instruct-v0:2 | mistral.mistral-7b-instruct-v0:2 | 155.48 | 0.081121 |
| mistral.mixtral-8x7b-instruct-v0:1 | mistral.mixtral-8x7b-instruct-v0:1 | 233.74 | 0.243235 |
| meta.llama2-13b-chat-v1 | meta.llama2-13b-chat-v1 | 245.77 | 0.401798 |
| amazon.titan-text-lite-v1 | amazon.titan-text-lite-v1 | 413.58 | 0.160314 |
| amazon.titan-text-express-v1 | amazon.titan-text-express-v1 | 309.98 | 0.431146 |
| anthropic.claude-3-sonnet-20240229-v1:0 | anthropic.claude-3-sonnet-20240229-v1:0 | 827.07 | 1.996164 |
| anthropic.claude-3-haiku-20240307-v1:0 | anthropic.claude-3-haiku-20240307-v1:0 | 243.72 | 0.163782 |
| cohere.command-text-v14 | cohere.command-text-v14 | 351.81 | 0.806398 |
| cohere.command-light-text-v14 | cohere.command-light-text-v14 | 406.41 | 0.164374 |
| ai21.j2-mid-v1 | ai21.j2-mid-v1 | 159.68 | 6.684875 |
| ai21.j2-ultra-v1 | ai21.j2-ultra-v1 | 206.63 | 10.041249 |




## Per instance results

The following table provides the best combinations for running inference for different sizes prompts on different instance types. The following dataset(s) were used for this test: `2wikimqa_e.jsonl`, `2wikimqa.jsonl`, `hotpotqa_e.jsonl`, `hotpotqa.jsonl`, `narrativeqa.jsonl`, `triviaqa_e.jsonl`, `triviaqa.jsonl`.

|Dataset   | Instance type   | Recommendation   |
|---|---|---|
|`payload_en_1-500.jsonl`|`ai21.j2-mid-v1`|The best option for staying within a latency budget of `2 seconds` on a `ai21.j2-mid-v1` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 0.43 seconds`, for an `average prompt size of 249 tokens` and `completion size of 24 tokens` with `273 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`ai21.j2-mid-v1`|The best option for staying within a latency budget of `2 seconds` on a `ai21.j2-mid-v1` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 0.55 seconds`, for an `average prompt size of 1376 tokens` and `completion size of 20 tokens` with `188 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`ai21.j2-mid-v1`|The best option for staying within a latency budget of `2 seconds` on a `ai21.j2-mid-v1` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 0.69 seconds`, for an `average prompt size of 2124 tokens` and `completion size of 25 tokens` with `151 transactions/minute`.|
|`payload_en_3000-4000.jsonl`|`ai21.j2-mid-v1`|The best option for staying within a latency budget of `2 seconds` on a `ai21.j2-mid-v1` for the `payload_en_3000-4000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 0.73 seconds`, for an `average prompt size of 2952 tokens` and `completion size of 20 tokens` with `151 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`ai21.j2-mid-v1`|The best option for staying within a latency budget of `2 seconds` on a `ai21.j2-mid-v1` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 0.49 seconds`, for an `average prompt size of 800 tokens` and `completion size of 24 tokens` with `235 transactions/minute`.|
|`payload_en_1-500.jsonl`|`ai21.j2-ultra-v1`|The best option for staying within a latency budget of `2 seconds` on a `ai21.j2-ultra-v1` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 0.53 seconds`, for an `average prompt size of 249 tokens` and `completion size of 24 tokens` with `220 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`ai21.j2-ultra-v1`|The best option for staying within a latency budget of `2 seconds` on a `ai21.j2-ultra-v1` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.03 seconds`, for an `average prompt size of 1376 tokens` and `completion size of 27 tokens` with `127 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`ai21.j2-ultra-v1`|The best option for staying within a latency budget of `2 seconds` on a `ai21.j2-ultra-v1` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.0 seconds`, for an `average prompt size of 2124 tokens` and `completion size of 18 tokens` with `118 transactions/minute`.|
|`payload_en_3000-4000.jsonl`|`ai21.j2-ultra-v1`|The best option for staying within a latency budget of `2 seconds` on a `ai21.j2-ultra-v1` for the `payload_en_3000-4000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 0.98 seconds`, for an `average prompt size of 2952 tokens` and `completion size of 17 tokens` with `116 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`ai21.j2-ultra-v1`|The best option for staying within a latency budget of `2 seconds` on a `ai21.j2-ultra-v1` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 0.6 seconds`, for an `average prompt size of 800 tokens` and `completion size of 22 tokens` with `192 transactions/minute`.|
|`payload_en_1-500.jsonl`|`amazon.titan-text-express-v1`|The best option for staying within a latency budget of `2 seconds` on a `amazon.titan-text-express-v1` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.05 seconds`, for an `average prompt size of 255 tokens` and `completion size of 25 tokens` with `111 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`amazon.titan-text-express-v1`|The best option for staying within a latency budget of `2 seconds` on a `amazon.titan-text-express-v1` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.32 seconds`, for an `average prompt size of 1382 tokens` and `completion size of 20 tokens` with `82 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`amazon.titan-text-express-v1`|The best option for staying within a latency budget of `2 seconds` on a `amazon.titan-text-express-v1` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.48 seconds`, for an `average prompt size of 2130 tokens` and `completion size of 18 tokens` with `95 transactions/minute`.|
|`payload_en_3000-4000.jsonl`|`amazon.titan-text-express-v1`|The best option for staying within a latency budget of `2 seconds` on a `amazon.titan-text-express-v1` for the `payload_en_3000-4000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.64 seconds`, for an `average prompt size of 2958 tokens` and `completion size of 17 tokens` with `77 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`amazon.titan-text-express-v1`|The best option for staying within a latency budget of `2 seconds` on a `amazon.titan-text-express-v1` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 0.7 seconds`, for an `average prompt size of 806 tokens` and `completion size of 5 tokens` with `163 transactions/minute`.|
|`payload_en_1-500.jsonl`|`amazon.titan-text-lite-v1`|The best option for staying within a latency budget of `2 seconds` on a `amazon.titan-text-lite-v1` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.39 seconds`, for an `average prompt size of 255 tokens` and `completion size of 39 tokens` with `82 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`amazon.titan-text-lite-v1`|The best option for staying within a latency budget of `2 seconds` on a `amazon.titan-text-lite-v1` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.7 seconds`, for an `average prompt size of 1382 tokens` and `completion size of 17 tokens` with `61 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`amazon.titan-text-lite-v1`|The best option for staying within a latency budget of `2 seconds` on a `amazon.titan-text-lite-v1` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.92 seconds`, for an `average prompt size of 2130 tokens` and `completion size of 11 tokens` with `77 transactions/minute`.|
|`payload_en_3000-4000.jsonl`|`amazon.titan-text-lite-v1`|This experiment did not find any combination of concurrency level and other configuration settings that could provide a response within a latency budget of `2 seconds` on a `amazon.titan-text-lite-v1` for the `payload_en_3000-4000.jsonl` dataset.|
|`payload_en_500-1000.jsonl`|`amazon.titan-text-lite-v1`|The best option for staying within a latency budget of `2 seconds` on a `amazon.titan-text-lite-v1` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.07 seconds`, for an `average prompt size of 806 tokens` and `completion size of 4 tokens` with `108 transactions/minute`.|
|`payload_en_1-500.jsonl`|`anthropic.claude-3-haiku-20240307-v1:0`|The best option for staying within a latency budget of `2 seconds` on a `anthropic.claude-3-haiku-20240307-v1:0` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 0.48 seconds`, for an `average prompt size of 272 tokens` and `completion size of 30 tokens` with `225 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`anthropic.claude-3-haiku-20240307-v1:0`|The best option for staying within a latency budget of `2 seconds` on a `anthropic.claude-3-haiku-20240307-v1:0` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.02 seconds`, for an `average prompt size of 1541 tokens` and `completion size of 54 tokens` with `107 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`anthropic.claude-3-haiku-20240307-v1:0`|The best option for staying within a latency budget of `2 seconds` on a `anthropic.claude-3-haiku-20240307-v1:0` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.12 seconds`, for an `average prompt size of 2379 tokens` and `completion size of 59 tokens` with `88 transactions/minute`.|
|`payload_en_3000-4000.jsonl`|`anthropic.claude-3-haiku-20240307-v1:0`|The best option for staying within a latency budget of `2 seconds` on a `anthropic.claude-3-haiku-20240307-v1:0` for the `payload_en_3000-4000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.23 seconds`, for an `average prompt size of 3281 tokens` and `completion size of 61 tokens` with `90 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`anthropic.claude-3-haiku-20240307-v1:0`|The best option for staying within a latency budget of `2 seconds` on a `anthropic.claude-3-haiku-20240307-v1:0` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.03 seconds`, for an `average prompt size of 887 tokens` and `completion size of 63 tokens` with `105 transactions/minute`.|
|`payload_en_1-500.jsonl`|`anthropic.claude-3-sonnet-20240229-v1:0`|The best option for staying within a latency budget of `2 seconds` on a `anthropic.claude-3-sonnet-20240229-v1:0` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 1.4 seconds`, for an `average prompt size of 272 tokens` and `completion size of 50 tokens` with `42 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`anthropic.claude-3-sonnet-20240229-v1:0`|This experiment did not find any combination of concurrency level and other configuration settings that could provide a response within a latency budget of `2 seconds` on a `anthropic.claude-3-sonnet-20240229-v1:0` for the `payload_en_1000-2000.jsonl` dataset.|
|`payload_en_2000-3000.jsonl`|`anthropic.claude-3-sonnet-20240229-v1:0`|This experiment did not find any combination of concurrency level and other configuration settings that could provide a response within a latency budget of `2 seconds` on a `anthropic.claude-3-sonnet-20240229-v1:0` for the `payload_en_2000-3000.jsonl` dataset.|
|`payload_en_3000-4000.jsonl`|`anthropic.claude-3-sonnet-20240229-v1:0`|This experiment did not find any combination of concurrency level and other configuration settings that could provide a response within a latency budget of `2 seconds` on a `anthropic.claude-3-sonnet-20240229-v1:0` for the `payload_en_3000-4000.jsonl` dataset.|
|`payload_en_500-1000.jsonl`|`anthropic.claude-3-sonnet-20240229-v1:0`|The best option for staying within a latency budget of `2 seconds` on a `anthropic.claude-3-sonnet-20240229-v1:0` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 1.19 seconds`, for an `average prompt size of 887 tokens` and `completion size of 62 tokens` with `49 transactions/minute`.|
|`payload_en_1-500.jsonl`|`cohere.command-light-text-v14`|The best option for staying within a latency budget of `2 seconds` on a `cohere.command-light-text-v14` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 0.85 seconds`, for an `average prompt size of 249 tokens` and `completion size of 29 tokens` with `101 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`cohere.command-light-text-v14`|The best option for staying within a latency budget of `2 seconds` on a `cohere.command-light-text-v14` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.11 seconds`, for an `average prompt size of 1376 tokens` and `completion size of 24 tokens` with `109 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`cohere.command-light-text-v14`|The best option for staying within a latency budget of `2 seconds` on a `cohere.command-light-text-v14` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 1.71 seconds`, for an `average prompt size of 2124 tokens` and `completion size of 42 tokens` with `43 transactions/minute`.|
|`payload_en_3000-4000.jsonl`|`cohere.command-light-text-v14`|This experiment did not find any combination of concurrency level and other configuration settings that could provide a response within a latency budget of `2 seconds` on a `cohere.command-light-text-v14` for the `payload_en_3000-4000.jsonl` dataset.|
|`payload_en_500-1000.jsonl`|`cohere.command-light-text-v14`|The best option for staying within a latency budget of `2 seconds` on a `cohere.command-light-text-v14` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 0.82 seconds`, for an `average prompt size of 800 tokens` and `completion size of 14 tokens` with `113 transactions/minute`.|
|`payload_en_1-500.jsonl`|`cohere.command-text-v14`|The best option for staying within a latency budget of `2 seconds` on a `cohere.command-text-v14` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 1`. A concurrency level of 1 achieves an `average latency of 0.87 seconds`, for an `average prompt size of 249 tokens` and `completion size of 23 tokens` with `69 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`cohere.command-text-v14`|The best option for staying within a latency budget of `2 seconds` on a `cohere.command-text-v14` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.81 seconds`, for an `average prompt size of 1376 tokens` and `completion size of 36 tokens` with `96 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`cohere.command-text-v14`|The best option for staying within a latency budget of `2 seconds` on a `cohere.command-text-v14` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.71 seconds`, for an `average prompt size of 2124 tokens` and `completion size of 28 tokens` with `84 transactions/minute`.|
|`payload_en_3000-4000.jsonl`|`cohere.command-text-v14`|The best option for staying within a latency budget of `2 seconds` on a `cohere.command-text-v14` for the `payload_en_3000-4000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.83 seconds`, for an `average prompt size of 2952 tokens` and `completion size of 23 tokens` with `74 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`cohere.command-text-v14`|This experiment did not find any combination of concurrency level and other configuration settings that could provide a response within a latency budget of `2 seconds` on a `cohere.command-text-v14` for the `payload_en_500-1000.jsonl` dataset.|
|`payload_en_1-500.jsonl`|`meta.llama2-13b-chat-v1`|The best option for staying within a latency budget of `2 seconds` on a `meta.llama2-13b-chat-v1` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 0.76 seconds`, for an `average prompt size of 249 tokens` and `completion size of 25 tokens` with `149 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`meta.llama2-13b-chat-v1`|The best option for staying within a latency budget of `2 seconds` on a `meta.llama2-13b-chat-v1` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 0.71 seconds`, for an `average prompt size of 1376 tokens` and `completion size of 13 tokens` with `171 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`meta.llama2-13b-chat-v1`|The best option for staying within a latency budget of `2 seconds` on a `meta.llama2-13b-chat-v1` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.05 seconds`, for an `average prompt size of 2124 tokens` and `completion size of 19 tokens` with `116 transactions/minute`.|
|`payload_en_3000-4000.jsonl`|`meta.llama2-13b-chat-v1`|The best option for staying within a latency budget of `2 seconds` on a `meta.llama2-13b-chat-v1` for the `payload_en_3000-4000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.36 seconds`, for an `average prompt size of 2952 tokens` and `completion size of 22 tokens` with `87 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`meta.llama2-13b-chat-v1`|The best option for staying within a latency budget of `2 seconds` on a `meta.llama2-13b-chat-v1` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 0.59 seconds`, for an `average prompt size of 800 tokens` and `completion size of 14 tokens` with `198 transactions/minute`.|
|`payload_en_1-500.jsonl`|`mistral.mistral-7b-instruct-v0:2`|The best option for staying within a latency budget of `2 seconds` on a `mistral.mistral-7b-instruct-v0:2` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 0.33 seconds`, for an `average prompt size of 256 tokens` and `completion size of 24 tokens` with `337 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`mistral.mistral-7b-instruct-v0:2`|The best option for staying within a latency budget of `2 seconds` on a `mistral.mistral-7b-instruct-v0:2` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 0.49 seconds`, for an `average prompt size of 1383 tokens` and `completion size of 26 tokens` with `246 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`mistral.mistral-7b-instruct-v0:2`|The best option for staying within a latency budget of `2 seconds` on a `mistral.mistral-7b-instruct-v0:2` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 0.61 seconds`, for an `average prompt size of 2131 tokens` and `completion size of 32 tokens` with `184 transactions/minute`.|
|`payload_en_3000-4000.jsonl`|`mistral.mistral-7b-instruct-v0:2`|The best option for staying within a latency budget of `2 seconds` on a `mistral.mistral-7b-instruct-v0:2` for the `payload_en_3000-4000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 0.75 seconds`, for an `average prompt size of 2959 tokens` and `completion size of 34 tokens` with `148 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`mistral.mistral-7b-instruct-v0:2`|The best option for staying within a latency budget of `2 seconds` on a `mistral.mistral-7b-instruct-v0:2` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 0.64 seconds`, for an `average prompt size of 807 tokens` and `completion size of 49 tokens` with `172 transactions/minute`.|
|`payload_en_1-500.jsonl`|`mistral.mixtral-8x7b-instruct-v0:1`|The best option for staying within a latency budget of `2 seconds` on a `mistral.mixtral-8x7b-instruct-v0:1` for the `payload_en_1-500.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 0.74 seconds`, for an `average prompt size of 256 tokens` and `completion size of 25 tokens` with `146 transactions/minute`.|
|`payload_en_1000-2000.jsonl`|`mistral.mixtral-8x7b-instruct-v0:1`|The best option for staying within a latency budget of `2 seconds` on a `mistral.mixtral-8x7b-instruct-v0:1` for the `payload_en_1000-2000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.09 seconds`, for an `average prompt size of 1383 tokens` and `completion size of 32 tokens` with `120 transactions/minute`.|
|`payload_en_2000-3000.jsonl`|`mistral.mixtral-8x7b-instruct-v0:1`|The best option for staying within a latency budget of `2 seconds` on a `mistral.mixtral-8x7b-instruct-v0:1` for the `payload_en_2000-3000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 0.88 seconds`, for an `average prompt size of 2131 tokens` and `completion size of 25 tokens` with `138 transactions/minute`.|
|`payload_en_3000-4000.jsonl`|`mistral.mixtral-8x7b-instruct-v0:1`|The best option for staying within a latency budget of `2 seconds` on a `mistral.mixtral-8x7b-instruct-v0:1` for the `payload_en_3000-4000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.23 seconds`, for an `average prompt size of 2959 tokens` and `completion size of 24 tokens` with `104 transactions/minute`.|
|`payload_en_500-1000.jsonl`|`mistral.mixtral-8x7b-instruct-v0:1`|The best option for staying within a latency budget of `2 seconds` on a `mistral.mixtral-8x7b-instruct-v0:1` for the `payload_en_500-1000.jsonl` dataset is a `concurrency level of 2`. A concurrency level of 2 achieves an `average latency of 1.32 seconds`, for an `average prompt size of 807 tokens` and `completion size of 51 tokens` with `87 transactions/minute`.|

## Plots

The following plots provide insights into the results from the different experiments run.

![Error rates for different concurrency levels and instance types](error_rates.png)

![Tokens vs latency for different concurrency levels and instance types](tokens_vs_latency.png)

![Concurrency Vs latency for different instance type for selected dataset](concurrency_vs_inference_latency.png)
