# Benchmarking for Amazon SageMaker JumpStart models

[Amazon SageMaker JumpStart](https://aws.amazon.com/sagemaker/jumpstart/getting-started/) offers state-of-the-art foundation models for use cases such as content writing, code generation, question answering, copywriting, summarization, classification, information retrieval, and more. A key challenge with FMs is the ability to benchmark their performance in terms of inference latency, throughput and cost so as to determine which model running with what combination of the following is the most optimal option for a given business use-case:

- Inference container and parameters (for example: batch size, degree of tensor parallelism, instance type/size etc.)
- Runtime inference parameters (for example temperature, top_p etc.)
- Choice of hardware (for example: `p4d`, `Inf2`, `g5`)  on different size types, for example 24x/48x etc.)

#### This repository provides a test harness for doing this benchmarking (note that this is different from benchmarking accuracy of a particular model for a given use-case).

### Instructions

#### Step 1: Configure your AWS Account + Config.yaml File

1. To configure your AWS account in your IDE environment, use the command 'aws configure' in the terminal and enter your aws account's access keys to get authenticated.

2. **Config.yml**: Your config.yml file will contain information about the follows:

    - **AWS account information**: Here, add the region you are using your account in, for example, 'us-east-1' for north virginia. Also, add the 'arn' for your sagemaker execution role with the right permissions, so you can deploy, invoke and access certain models within via the sagemaker jumpstart route.

    - **Directory Paths**: This contains directory paths for where your model details, for example the configuration information for your model that you will deploy, and prompts that it will be invoked on be stored. This is flexible based on where you store the files.

    - **Dataset**: Add the name of your dataset which will lie in the data/dataset folder. This will contain the prompts you want to invoke your jumpstart to be deployed on. This dataset can be of various kinds, for text generation, summarization, question answering, based on what you want your models to be benchmarked on. Your dataset in our repository will be segregated into sizes of prompts for better insights into what the best prompt length is for your use case based on latency and other key metric requirements.

    - **prompt**: Here, you may or may not have this portion but if your dataset contains for example multiple languages, you can use this section to configure filtering the data, for example have the launguage be english, and configure the minimum and maximum length tokens for these prompts that the deployed models will invoke.

    - **Prompt Template**: The prompt template will be based on the kind of model you use. In this repo, we used the LLaMa-2 jumpstart models, so the prompt template is formated and used from the 'prompt_template.txt' file. If you want to change it based on the model you deploy, you can simply modify this file to adapt to the new prompt template. This file is stored in data/prompts/prompt_template.txt

    - **Concurrency Levels**: Based on your business use case, you can set concurrency levels for each instance, and test your model performance based on latency requirements. Say you want to test for "how many requests of sizes 2000-3000 token length can my LLaMa-2-13b on an inf2.48x large take within 5 seconds?", in this case, you can measure each concurrency level through this config file and then use it while running inferences on your deployed model endpoints from SageMaker Jumpstart.

    - **Inference Parameters**: These are the parameters that your model will give inference at during invocation. These are flexible to change, based on use case to use case to benchmark different metrics. Here we are including do_sample, top_p, top_k, max_new_tokens, temperature, and truncate values.

    - **Model Configurations**: This will contain the models you want to deploy while running the first notebook (0_deploy_model.ipynb). This is flexible to change based on what you need the models to be deployed on specific configurations, such as the specific 'OPTION_TENSOR_PARALLEL_DEGREE' you might need, or the 'OPTION_N_POSITIONS', 'instance_type' or the 'image_uri'. You can feel free to change these jumpstart models to different offerings all across sagemaker jumpstart and use the models you want to use in this test harness for the best use of flexibility, cost and performance.

### Initialize the globals.py file to use instances across the test harness

```
import os
from enum import Enum
from pathlib import Path
CONFIG_FILE: str = "config.yml"
DATA_DIR: str = "data"
PROMPTS_DIR = os.path.join(DATA_DIR, "prompts")
METRICS_DIR = os.path.join(DATA_DIR, "metrics")
MODELS_DIR = os.path.join(DATA_DIR, "models")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
DIR_LIST = [DATA_DIR, PROMPTS_DIR, METRICS_DIR, MODELS_DIR, DATASET_DIR]
TOKENIZER_DIR = 'llama2_tokenizer'
_ = list(map(lambda x: os.makedirs(x, exist_ok=True), DIR_LIST))
ENDPOINT_LIST_FPATH:str = os.path.join(MODELS_DIR, "endpoints.json")
REQUEST_PAYLOAD_FPATH:str = os.path.join(PROMPTS_DIR, "payload.jsonl")
RESULTS_FPATH:str = os.path.join(METRICS_DIR, "results.csv")
class TRUNCATE_POLICY(str, Enum):
    AT_PROMPT_TOKEN_LENGTH = 'at-prompt-token-length'
```

- We will pygmentize the ***global instances*** in notebooks for variables above including contents of the config file, the prompts, metrics, tokenizer and dataset directories. This is flexible to change based on where your files and data reside.

### Install the LLaMa-2 Tokenizer

1. Download the LLaMA 2 Tokenzier from https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main 
   and place the files into a directory named `llama2_tokenizer` in the same 
   directory as this notebook.

***If you do not have access to this repository, fill our the Meta form to get approval to download the tokenizer files. ***
Form link: - https://l.facebook.com/l.php?u=https%3A%2F%2Fgithub.com%2Ffacebookresearch%2Fllama&h=AT3hovLGEdz6VUvMRDlyOVw1DnnQM_MikTt01t3pkWzFPcP6GaiQLDoMKRnONq7qXIGY_FN2rRhSHt30N-Nhmlo6vnm4sIs3SwuUOJiAf_-_s8SmzPSCwwUKh9zxx_DvbrNc_t2owGO0Anu_1ENlRY3jDhLmcg

#### Now that you have the config.yml, globals.py, datasets and llama tokenizer in place, we can start running the three main notebooks: 

#### Set up the environment using the code in the setup notebook: `0_setup.ipynb` 
(https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/0_setup.ipynb) 

#### 1. Navigate to the file: `1_deploy_model.ipynb` 

- This is where we are deploying the models, where you have the ability to configure the model parameters and configurations. We are using ***LLaMa-2-7b, 13b, and chat models*** that are supported by Neuron now on Sagemaker Jumpstart, but you can use any models offered on jumpstart now. (https://github.com/aws-samples jumpstart-models-benchmarking-test-harness/blob/main/1_deploy_model.ipynb)

- Install all of the requirements from the requirements.txt file and most of the libraries to import: 

```
ipywidgets==8.1.1
sagemaker==2.203.0
transformers==4.36.2
pandas==2.1.4
datasets==2.16.1
seaborn==0.13.1
```

- ***Initialize the global variables from the globals.py*** file as well as load the aws sagemaker execution role and region into the logger 

- Have an async function system to deploy models from the model configs in config.yml concurrently. To double check, once you start the deployment, you can view the models being created and deployed together on your aws console.

- Once the jumpstart models are deployed, you can view the model configurations and environment information in the data/models/endpoints.json file.

#### 2.  Now run the notebook: `2_generate_data.ipynb`: 

- In this file, we will load the dataset, fit the prompt template with the dataset we have. Let's take a look at some steps within this notebook as you run it. You can run this notebook in parallel while the models are being deployed for the interest of time. (https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/2_generate_data.ipynb)

- Initialize the requirements, libraries to be imported, and global variables along with the logger info.

- Extract the prompt template file.txt from the appropriate file path. This would show the prompt template that you are using which will change based on the prompt format required by the specific jumpstart models that you are deploying. In the case where you are deploying more that a single kind of model, you can create multiple prompt template files and associate it in this ntoebook. 

- Once you have loaded your data files from the file and converted it into a dataframe in this notebook, you can fit the data into the prompt template you have for your specific jumpstart model type.

- Write the prompts into a json file for further processing during invoking the deployed jumpstart models for inference and benchmarking.

```
# convert the prompts into payload we can send to the model
def construct_request_payload(row, config: Dict) -> Dict:
    parameters = copy.deepcopy(config['inference_parameters'])
    if parameters['truncate'] == TRUNCATE_POLICY.AT_PROMPT_TOKEN_LENGTH:
        parameters['truncate'] = row['prompt_len']
    return dict(inputs=row['prompt']['prompt'], parameters=parameters)

df_filtered['request'] = df_filtered.apply(lambda r: construct_request_payload(r, config), axis=1)

logger.info(f"payload request entry looks like this -> {json.dumps(df_filtered['request'].iloc[0], indent=2)}")
```


#### 3. Now run the notebook: '3_run_inference.ipynb'. While you run this notebook, there are certain tasks being handled: 

- Notebook link: https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/3_run_inference.ipynb

- Here we aggregate the jumpstart models that we deployed asynchronously, along with the dataset we fit into a payload json file based on the data you might have. Now:

- We extract the active endpoints we deployed, initialize a 'payload list' which we will iterate through concurrently on each endpoint to generate inference for benchmarking purposes.

- We will run all requests to the endpoints we deploy concurrently that we will specify in the config.yml file. For example, in this case we are invoking the llama-2 13b on an inf2.48x and an inf2.24x with a concurrency level of 1 versus 2 versus 4. In the same way you can experiment with multiple configurations.

First, we create a predictor object for our endpoints that we have deployed. Then we asynchronously call the async_get_inference on the get_inference function followed by gather all these async tasks and running them on 'async_get_all_inferences'.

- At the time of the inference, we record all metrics, such as inference latency, transactions per second, token throughput, and other payload metrics we track such as top_k, top_p, temperature and so on. Here is how we are calculating all of our metrics in a series of async. functions:

```
errors = [r for r in responses if r['completion'] is None]
    successes = len(chunk) - len(errors)
    all_prompts_token_count = safe_sum([r['prompt_tokens'] for r in responses])
    prompt_token_throughput = round(all_prompts_token_count / elapsed_async, 2)
    prompt_token_count_mean = safe_div(all_prompts_token_count, successes)
    all_completions_token_count = safe_sum([r['completion_tokens'] for r in responses])
    completion_token_throughput = round(all_completions_token_count / elapsed_async, 2)
    completion_token_count_mean = safe_div(all_completions_token_count, successes)
    transactions_per_second = round(successes / elapsed_async, 2)
    transactions_per_minute = int(transactions_per_second * 60)
    latency_mean = safe_div(safe_sum([r['latency'] for r in responses]), successes)
```

#### Once all of the responses are recorded, we store the responses in the form of a dataframe or a csv with a couple of columns of interests as given below:

```
cols_of_interest = ['experiment_name', 
                    'instance_type',
                    'endpoint.EndpointName',
                    'model_config.ModelName',
                    'model_config.PrimaryContainer.Image',   
                    'model_config.PrimaryContainer.ModelDataSource.S3DataSource.S3Uri',
                    'model_config.PrimaryContainer.Environment.OPTION_DTYPE',
                    'model_config.PrimaryContainer.Environment.OPTION_MAX_ROLLING_BATCH_SIZE',
                    'model_config.PrimaryContainer.Environment.OPTION_NEURON_OPTIMIZE_LEVEL',
                    'model_config.PrimaryContainer.Environment.OPTION_N_POSITIONS',
                    'model_config.PrimaryContainer.Environment.OPTION_ROLLING_BATCH',
                    'model_config.PrimaryContainer.Environment.OPTION_TENSOR_PARALLEL_DEGREE',
                    'model_config.PrimaryContainer.Environment.SAGEMAKER_MODEL_SERVER_WORKERS']
```

You can feel flexible in terms of generating and using new columns based on what you want to track while benchmarking your sagemaker jumpstart model.

All results are stored in the results.csv file in the data/metrics/results.csv file path.

#### 4. Lastly, run the '4_model_metric_analysis.ipynb' to record all of the metrics from the model inference benchmarking that we did and visualize the findings:

- Notebook link: https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/4_model_metric_analysis.ipynb

- In this file, we read the contents of the results.csv file and focus on generating visualizations using the following columns we have:

```
Index(['endpoint_name', 'prompt', 'do_sample', 'temperature', 'top_p', 'top_k',
       'max_new_tokens', 'truncate', 'completion', 'prompt_tokens',
       'completion_tokens', 'latency', 'tps', 'token_throughput',
       'EndpointName', 'ModelName', 'Image', 'S3Uri', 'OPTION_DTYPE',
       'OPTION_MAX_ROLLING_BATCH_SIZE', 'OPTION_NEURON_OPTIMIZE_LEVEL',
       'OPTION_N_POSITIONS', 'OPTION_ROLLING_BATCH',
       'OPTION_TENSOR_PARALLEL_DEGREE', 'SAGEMAKER_MODEL_SERVER_WORKERS',
       'concurrency'],
      dtype='object')
```

#### - Visualize the prompt tokens and how many prompts are of different context lengths:

This will help us visualize how the benchmarks change in terms of when the prompt length/context length increases/decreases.

- We can also view how the latency is progressed as the prompt lengths etc, increase: 


<code>sns.ecdfplot(data=df, x="latency")</code>

***Furthermore, we can get a correlation between the concurrency of the specific model endpoint during invocation, the latency of the inference and the prompt tokens that is was processing during this time:***

<code>sns.scatterplot(data=df, x="prompt_tokens", y="latency", hue="concurrency")</code>

#### - Visualize the Effect of token length on inference latency:

```
plt.xlabel("Prompt length (tokens)")
plt.ylabel("Latency (seconds)")
plt.title(f"Effect of token length on inference latency")
sns.boxplot(data=df_prompt_len_and_latency, x="label", y="latency", hue="concurrency", order=labels)
```

In this notebook, you can visualize how the latency increases or decreases, or remains static as the prompt length increase at a specific concurrency level. You can view this for multiple jumpstart models that you deploy.





