# Benchmark models deployed on different AWS Generative AI services

`FMBench` comes packaged with configuration files for benchmarking models on different AWS Generative AI services. 

## Full list of benchmarked models


| Model                           | Amazon EC2                     | Amazon SageMaker                           | Amazon Bedrock                     |
|:--------------------------------|:-------------------------------|:-------------------------------------------|:-----------------------------------|
| **Deepseek-R1 distilled**        | g6e                           | g6e                                           |                            |
| **Llama3.3-70b instruct**        |                               |                                           | On-demand                           |
| **Qwen2.5-72b**                  | g5, g6e                       |                                           |                                    |
| **Amazon Nova**                  |                               |                                           | On-demand                          |
| **Anthropic Claude-3 Sonnet**    |                               |                                           | On-demand, provisioned             |
| **Anthropic Claude-3 Haiku**     |                               |                                           | On-demand                          |
| **Mistral-7b-instruct**          | inf2, trn1                     | g4dn, g5, p3, p4d, p5                       | On-demand                          |
| **Mistral-7b-AWQ**               |                               | p5                                        |                                    |
| **Mixtral-8x7b-instruct**        |                               |                                           | On-demand                          |
| **Llama3.2-1b instruct**         | g5                            |                                           |                                    |
| **Llama3.2-3b instruct**         | g5                            |                                           |                                    |
| **Llama3.1-8b instruct**         | g5, p4d, p4de, p5, p5e, g6e, g6, inf2, trn1        | g4dn, g5, p3, inf2, trn1                     | On-demand                          |
| **Llama3.1-70b instruct**        | p4d, p4de, p5, p5e, g6e, g5, inf2, trn1            | inf2, trn1                                 | On-demand                          |
| **Llama3-8b instruct**           | g5, g6e, inf2, trn1, c8g      | g4dn, g5, p3, inf2, trn1, p4d, p5e             | On-demand                          |
| **Llama3-70b instruct**          | g5                            | g4dn, g5, p3, inf2, trn1, p4d                 | On-demand                          |
| **Llama2-13b chat**              |                               | g4dn, g5, p3, inf2, trn1, p4d                 | On-demand                          |
| **Llama2-70b chat**              |                               | g4dn, g5, p3, inf2, trn1, p4d                 | On-demand                          |
| **NousResearch-Hermes-70b**      |                               | g5, inf2, trn1                            | On-demand                          |
| **Amazon Titan text lite**       |                               |                                           | On-demand                          |
| **Amazon Titan text express**    |                               |                                           | On-demand                          |
| **Cohere Command text**          |                               |                                           | On-demand                          |
| **Cohere Command light text**    |                               |                                           | On-demand                          |
| **AI21 J2 Mid**                  |                               |                                           | On-demand                          |
| **AI21 J2 Ultra**                |                               |                                           | On-demand                          |
| **Gemma-2b**                     |                               | g4dn, g5, p3                                |                                    |
| **Phi-3-mini-4k-instruct**       |                               | g4dn, g5, p3                                |                                    |
| **distilbert-base-uncased**      |                               | g4dn, g5, p3                                |                                    |