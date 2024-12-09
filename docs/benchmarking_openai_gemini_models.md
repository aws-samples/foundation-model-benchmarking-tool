# Benchmark OpenAI and Gemini models

This feature enables users to benchmark _external_ models such as OpenAI and Gemini models on `FMBench`. Current models that are tested with this feature are: `gpt-4o`, `gpt-4o-mini`, `gemini-1.5-pro` and `gemini-1.5-flash`.

Choose any configuration file from the [`external`](https://github.com/aws-samples/foundation-model-benchmarking-tool/tree/main/src/fmbench/configs/external) folder and either run these directly or use them as templates for creating new config files specific to your use-case.

### Prerequisites

To benchmark an _external_ model, the configuration file requires a model provider provided **API Key** (such as an OpenAI key or a Gemini key). If you are benchmarking on an `EC2` instance, create an `openai_key.txt` or `gemini_key.txt` file in the `/tmp/fmbench-read/scripts/` directory depending on the API key that you would like to use. In the command below replace the `your_api_key` with your API key.

- For _OpenAI_ models: 
```
    echo your_api_key > /tmp/fmbench-read/scripts/openai_key.txt
``` 

- For _Gemini_ models: 
```
    echo your_api_key > /tmp/fmbench-read/scripts/gemini_key.txt
``` 

**_Note_**: If you are benchmarking on Amazon SageMaker, place the API key files in the `fmbench-read` bucket within the `scripts` directory.

### Run the benchmarking test

Once you have set up the API key in the configuration file, that is all. Run the test using the command below. This is an example for benchmarking the `OpenAI` models on FMBench.

```{.bash}
fmbench --config-file https://raw.githubusercontent.com/aws-samples/foundation-model-benchmarking-tool/main/src/fmbench/configs/external/openAI/config-openai-models.yml > fmbench.log 2>&1
```
