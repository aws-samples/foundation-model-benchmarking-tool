# Bring Your Own Dataset

By default `FMBench` uses the [`LongBench dataset`](https://github.com/THUDM/LongBench) dataset for testing the models, but this is not the only dataset you can test with. You may want to test with other datasets available on HuggingFace or use your own datasets for testing. 

## Hugging Face Data Preparation is now integrated within FMBench

FMBench now supports direct loading of Hugging Face datasets through a simple prefix mechanism. To use a Hugging Face dataset:

1. In your config file, prefix the dataset name with `hf:` in the `source_data_files` section:

    ```yaml
    source_data_files:
    - hf:derek-thomas/ScienceQA.jsonl
    ```

When FMBench encounters a dataset prefixed with `hf:`, it will:

- Automatically download the dataset from Hugging Face
- Convert it to the required JSON Lines format
- Handle both text and image datasets dynamically
- Store the processed data in either:
  - The S3 read bucket for cloud deployments
  - The `/tmp/fmbench-read/source_data/` directory for local runs

> **Note**: This requires a Hugging Face token to be configured in your environment for private or gated datasets.

## Using Custom Datasets

If you want to use your own dataset or a pre-processed dataset, you can:

- Provide the dataset path without the `hf:` prefix in the config:

    ```yaml
    source_data_files:
    - my-custom-dataset.jsonl
    ```

- Or, use the `[`bring_your_own_dataset`](./src/fmbench/bring_your_own_dataset.ipynb) notebook` to convert your custom dataset to JSON Lines format and upload it to the appropriate S3 bucket or local directory.

FMBench will use these files directly from the specified location without any preprocessing.

## Support for new Image and Text datasets

While you can use any hugging face dataset without pre processing, FMBench provides configuration files for running `llama3-2-11b-instruct`, `claude-3-sonnet`, `claude-3-5-sonnet` on the following image and text datasets:

1. Databricks dolly dataset: [config-llama-3-2-11b-databricks-dolly-15k.yml](https://github.com/aws-samples/foundation-model-benchmarking-tool/src/fmbench/configs/bedrock/config-llama-3-2-11b-databricks-dolly-15k.yml)
1. Multimodal ScienceQA dataset: [config-llama-3-2-11b-vision-instruct-scienceqa.yml](https://github.com/aws-samples/foundation-model-benchmarking-tool/src/fmbench/configs/multimodal/bedrock/config-llama-3-2-11b-vision-instruct-scienceqa.yml)
1. Multimodal marqo-GS-10M dataset: [config-llama-3-2-11b-vision-instruct-marqo-GS-10M.yml](https://github.com/aws-samples/foundation-model-benchmarking-tool/src/fmbench/configs/multimodal/bedrock/config-llama-3-2-11b-vision-instruct-marqo-GS-10M.yml)

## Support for Open-Orca dataset

Support for [Open-Orca](https://huggingface.co/datasets/Open-Orca/OpenOrca) dataset and corresponding prompts for Llama3, Llama2 and Mistral, see:

1. [bring_your_own_dataset.ipynb](https://github.com/aws-samples/foundation-model-benchmarking-tool/tree/main/src/fmbench/bring_your_own_dataset.ipynb)
1. [prompt templates](https://github.com/aws-samples/foundation-model-benchmarking-tool/tree/main/src/fmbench/prompt_template)
1. [Llama3 config file with OpenOrca](https://github.com/aws-samples/foundation-model-benchmarking-tool/tree/main/src/fmbench/configs/llama3/8b/config-llama3-8b-inf2-g5-byoe-w-openorca.yml)