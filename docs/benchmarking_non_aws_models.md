# Benchmark non AWS models on FMBench

This feature enables users to benchmark non AWS models on FMBench, such as OpenAI and Gemini models. Current models that are tested with this feature are: `gpt-4o`, `gpt-4o-mini`, `gemini-1.5-pro` and `gemini-1.5-flash`.

Choose any configuration file from the [`external`](https://github.com/aws-samples/foundation-model-benchmarking-tool/tree/main/src/fmbench/configs/external) folder and either run these directly or use them as templates for creating new config files specific to your use-case.

### Prerequisites

To benchmark a non AWS model, the configuration file requires an **API Key**. Mention your custom API key within the `inference_spec` section in the `experiments` within the configuration file. View an example below:

_Replace `<your-api_key>` with your actualy API key_

```{.yml}
inference_spec:
    split_input_and_parameters: no
    api_key: <your-api-key>
    parameter_set: external
    stream: True
```

### Run the benchmarking test

Once you have set up the API key in the configuration file, that is all. Run the test using the command below. This is an example for benchmarking the `OpenAI` models on FMBench.

```{.bash}
fmbench --config-file https://raw.githubusercontent.com/aws-samples/foundation-model-benchmarking-tool/main/src/fmbench/configs/external/openAI/config-openai-models.yml > fmbench.log 2>&1
```
