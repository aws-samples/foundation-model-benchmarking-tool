# Benchmark models deployed on different AWS Generative AI services

`FMBench` comes packaged with configuration files for benchmarking models on different AWS Generative AI services.

## Full list of benchmarked models

| Model                           | EC2 g5 | EC2 Inf2/Trn1 | SageMaker g4dn/g5/p3 | SageMaker Inf2 | SageMaker P4 | SageMaker P5 | Bedrock On-demand throughput | Bedrock provisioned throughput |
|:--------------------------------|:-------|:--------------|:---------------------|:---------------|:-------------|:-------------|:-----------------------------|:--------------------------------|
| **Anthropic Claude-3 Sonnet**   |        |               |                     |                |              |              | ✅                           | ✅                               |
| **Anthropic Claude-3 Haiku**    |        |               |                     |                |              |              | ✅                           |                                    |
| **Mistral-7b-instruct**          |        |               | ✅                   |                | ✅            | ✅           | ✅                           |                                    |
| **Mistral-7b-AWQ**               |        |               |                     |                |              | ✅           |                             |                                    |
| **Mixtral-8x7b-instruct**       |        |               |                     |                |              |              | ✅                           |                                    |
| **Llama3.1-8b instruct**         |        |               |                     |                |              |              | ✅                           |                                    |
| **Llama3.1-70b instruct**        |        |               |                     |                |              |              | ✅                           |                                    |
| **Llama3-8b instruct**           |  ✅      | ✅              | ✅                   | ✅             | ✅           | ✅           | ✅                           |                                    |
| **Llama3-70b instruct**          |  ✅      |               | ✅                   | ✅             | ✅           |              | ✅                           |                                    |
| **Llama2-13b chat**              |        |               | ✅                   | ✅             | ✅           |              | ✅                           |                                    |
| **Llama2-70b chat**              |        |               | ✅                   | ✅             | ✅           |              | ✅                           |                                    |
| **Amazon Titan text lite**       |        |               |                     |                |              |              | ✅                           |                                    |
| **Amazon Titan text express**    |        |               |                     |                |              |              | ✅                           |                                    |
| **Cohere Command text**          |        |               |                     |                |              |              | ✅                           |                                    |
| **Cohere Command light text**    |        |               |                     |                |              |              | ✅                           |                                    |
| **AI21 J2 Mid**                  |        |               |                     |                |              |              | ✅                           |                                    |
| **AI21 J2 Ultra**                |        |               |                     |                |              |              | ✅                           |                                    |
| **Gemma-2b**                     |        |               | ✅                   |                |              |              |                             |                                    |
| **Phi-3-mini-4k-instruct**       |        |               | ✅                   |                |              |              |                             |                                    |
| **distilbert-base-uncased**      |        |               | ✅                   |                |              |              |                             |                                    |

