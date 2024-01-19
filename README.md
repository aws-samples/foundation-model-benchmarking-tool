# Benchmarking for Amazon SageMaker JumpStart models and bring your own models

*Authors*: *Amit Arora*, *Madhur Prashant*

[Amazon SageMaker JumpStart](https://aws.amazon.com/sagemaker/jumpstart/getting-started/) offers state-of-the-art foundation models (FMs) for use cases such as content writing, code generation, question answering, copywriting, summarization, classification, information retrieval, and more. A key challenge with FMs is the ability to benchmark their performance in terms of inference latency, throughput and cost so as to determine which model running with what combination of the following is the most optimal option for a given use-case:

- Inference container and parameters (for example: batch size, degree of tensor parallelism etc.)
- Runtime inference parameters (for example temperature, top_p etc.)
- Choice of hardware (for example: `p4d`, `Inf2`, `g5` etc.)

## Solution overview

The solution presented in this repository provides a test harness for benchmarking not only Sagemaker Jumpstart FMs, but models out of Jumpstart where you can bring your own inference script and servings file, and run this repo as a test for inference latency, token throughput, concurrency metrics, and more. This README contains step by step instructions to run these tests.

In this repository, we are using the Llama JumpStart models (`13b and 70b`)(https://ai.meta.com/llama/) and the Hugging Face Llama [`13b-chat`](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) and [`70b-chat`](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) models. For both Jumpstart and non Jumpstart models, we are using the instance types below to test for metrics while answering high business value questions like: ***How many requests can a single endpoint with instance_type 'x' handle with a requirement of 'y' latency with a budget of $'z' for cost and performance efficiency?***

***View a portion of the visualization programmatically generated at the end of our benchmarking below:***

[ADD THE PLOT CHART HERE WITH A TWO LINER EXPLANATION]

**The instances that we are using in this repository, (but are not limited to these instance types) are:**

- [**p4d based instances**](https://aws.amazon.com/about-aws/whats-new/2020/12/introducing-amazon-sagemaker-ml-p4d-instances-for-highest-performance-ml-training-in-the-cloud/): `ml.p4d.24xlarge`
- [**g5 based instances**](https://aws.amazon.com/about-aws/whats-new/2022/01/sagemaker-training-ml-g5-instances/): `ml.g5.12xlarge`, `ml.g5.24xlarge`, `ml.g5.48xlarge`
- [**Inferentia based instances**](https://aws.amazon.com/about-aws/whats-new/2023/05/sagemaker-ml-inf2-ml-trn1-instances-model-deployment/): `ml.inf2.24xlarge`, `ml.inf2.48xlarge`

For the `bring your own model` option, we are supporting the latest versions of the DJL image uri that can be viewed in this [DJL documentation](https://docs.djl.ai/docs/serving/serving/docs/large_model_inference.html) and the [Hugging Face TGI container](https://huggingface.co/blog/sagemaker-huggingface-llm).


## Solution Prerequisites:

To run this open-source benchmarking harness for your specific use case, follow the prerequisites below to set up your environment before running the code:

1. **Llama 2 Tokenizer Requirements** : The use of this model is governed by the Meta license. In order to download the model weights and tokenizer, please visit the website and accept our License before requesting access here: [meta approval form](https://ai.meta.com/resources/models-and-libraries/llama-downloads/). Once you have been approved, please download the following files into the [llama2_tokenizer](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/tree/main/llama2_tokenizer) directory: 

    - `tokenizer_config.json`
    - `tokenizer.model`
    - `tokenizer.json`
    - `special_tokens_map.json`
    - `pytorch_model.bin.index.json`
    - `model.safetensors.index.json`
    - `generation_config.json`
    - `config.json`

***To access and download the Llama 2 files after being approved, view the [Hugging Face website](https://huggingface.co/meta-llama/Llama-2-7b/tree/main) for your model of use.***

2. **Data Ingestion** : Create a directory named "dataset" within the [data](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/tree/main/data) folder of this repository. 

- In our solution, we used `Q&A` datasets from the ***[`LongBench dataset`](https://github.com/THUDM/LongBench)***. 

    - Download the different files specified in the [LongBench dataset](https://github.com/THUDM/LongBench) into the `dataset` directory you created above. Here are the file names that you can download from [this website](https://github.com/THUDM/LongBench) to get started:

        - `2wikimqa`
        - `hotpotqa`
        - `narrativeqa`
        - `triviaqa`

    Once you have uploaded these files within your dataset folder in your [data directory](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/tree/main/data), complete the final prerequisite below.

3. **Hugging Face Auth Tokens For file access** : For some of the hugging face models that you use, a requirement would be to download model files from ***Hugging Face*** authenticated by your `HF Auth Token`. To define your auth token, create a file named `hf_token.txt` containing your Hugging Face auth token. 

    - `Instructions to access your `Auth token` can be [viewed here](https://huggingface.co/docs/hub/security-tokens). Once this is done, you will be able to download files without any errors. 

    - The ***hf_token.txt*** you define will be ignored as you publish your code publicly since the token is checked into the [.gitignore file](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/.gitignore), so that your auth token is not exposed to the public in the code repositories.

## Solution Design:

This solution design consists of 6 main steps. This includes:

1. Setting up your aws environment, configuring a [config.yml file](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/config.yml) that will contain information that will be loaded into all executable notebooks in our benchmarking process. 

2. Accessing different SageMaker JumpStart or non JumpStart models and deploying them seamlessly via this solution in your aws account.

3. All data will be generated to create payloads within your [data](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/tree/main/data) directory as mentioned in the prerequisites above. To make this benchmarking smoother and more granular in terms of the context token length that each model can take in, we will create payloads of different sizes for example, ***1-500 tokens, 500-1000 tokens, 1000-2000 tokens, 2000-3000 tokens and finally 3000-4000 tokens***. 

4. Run inferences using [this notebook](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/3_run_inference.ipynb) concurrently and asynchronously on each deployed endpoint with different combination of payload sizes and concurrency metrics

5. Track the inference latency and other metrics we will display via the [metrics analysis](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/4_model_metric_analysis.ipynb) notebook. 

6. Clean up your deployed endpoints using [this notebook](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/5_cleanup.ipynb)

***View the detailed walkthrough of each step below.***

### Step 1: Configure your Config.yml file

Navigate to [config.yaml file](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/config.yml), Configure your AWS account in your IDE environment, using `aws configure` in your command line interface, enter your access and secret access keys. 

***Enter your SageMaker execution role and region here:***

    ```
    aws:
      region: <your-aws-account-region-name>
      sagemaker_execution_role: <your-SageMaker-execution-role-name>
    ```
      Make sure your execution role has the right permissions.

This ***config.yml*** file consists of:

- Information on the sizes of your dataset payloads in the `datasets` section.

- ***Inference Parameters***: This is based on your model and requirements that can be viewed in the `inference_parameters` section of the file. In this example, we are using different ***Llama 2 Models*** - you can modify the parameters based on the specific model you are using.

- ***Model Configurations (experiments)*** : The `experiments` section of the file contains the model configurations for the SageMaker JumpStart and non JumpStart models to be deployed for benchmarking. This configuration consists of two main parts: configurations for ***SageMaker JumpStart models** & ***bring your own model***. 

**Here are two examples of model configurations for JumpStart and non JumpStart Models:**

1. **Configurations for the SageMaker JumpStart Model**: meta-textgeneration-llama-2-13b:

    ```
    - name: llama2-13b-g5.24xlarge-huggingface-pytorch-tgi-inference-2.0.1-tgi1.1.0
    model_id: meta-textgeneration-llama-2-13b
    model_version: "*"
    model_name: llama2-13b
    ep_name: llama-2-13b-g5-24xlarge
    instance_type: "ml.g5.24xlarge"
    image_uri: '763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.0.1-tgi1.1.0-gpu-py39-cu118-ubuntu20.04'
    deploy: no
    instance_count: 1
    deployment_script: jumpstart.py
    payload_files:
    - payload_en_1-500.jsonl
    - payload_en_500-1000.jsonl
    - payload_en_1000-2000.jsonl
    - payload_en_2000-3000.jsonl
    - payload_en_3000-4000.jsonl

    concurrency_levels:
    - 1
    - 2
    - 4
    accept_eula: true
    env:
    SAGEMAKER_PROGRAM: "inference.py"
    ENDPOINT_SERVER_TIMEOUT: "3600"
    MODEL_CACHE_ROOT: "/opt/ml/model"
    SAGEMAKER_ENV: "1"
    HF_MODEL_ID: "/opt/ml/model"
    MAX_INPUT_LENGTH: "4095"
    MAX_TOTAL_TOKENS: "4096"
    SM_NUM_GPUS: "4"
    SAGEMAKER_MODEL_SERVER_WORKERS: "1"
    ```

Here, we are configuring the `version of the model`, the `endpoint name`, `model_id` that needs to be deployed, `instance type` to be used, in this case: `ml.g5.24xlarge`, the `tgi image uri`, whether or not to deploy this given model, followed by an inference script ["jumpstart.py"](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/scripts/jumpstart.py) which supports the inference script for jumpstart models to deploy the model in our main [deploy notebook](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/1_deploy_model.ipynb). 

You can also set the payload files of different prompt sizes - for `1-500` tokens, `500-1000` tokens and so on until `4000` token length prompts. Lastly, ***set the concurrency levels, so the payloads will be combined with each concurrency level and the benchmarking will take place accordingly***. Include some final environment variables to use during model deployment such as the `SAGEMAKER_PROGRAM`, `ENDPOINT_SERVER_TIMEOUT`, and the rest above.

2. **Configurations for the Bring Your Own Model:**:

    ```
        # P4D Based Instance Model Configuration:
    - name: llama2-70b-p4d.24xlarge-tgi-inference-2.0.1-tgi0.9.3-gpu-py39-cu118
        model_id: meta-llama/Llama-2-70b-chat-hf
        model_version: "3.0.2"
        model_name: meta-llama-Llama-2-70b-hf
        ep_name: llama-2-70b-p4d-24xlarge    
        instance_type: "ml.p4d.24xlarge"    
        image_uri: 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.0.1-tgi0.9.3-gpu-py39-cu118-ubuntu20.04
        deploy: yes
        instance_count: 1
        deployment_script: p4d_hf_tgi.py
        payload_files:
        - payload_en_1-500.jsonl
        - payload_en_500-1000.jsonl
        - payload_en_1000-2000.jsonl
        - payload_en_2000-3000.jsonl
        - payload_en_3000-4000.jsonl
        concurrency_levels:
        - 1
        - 2
        - 4 
        accept_eula: true
        env:
        MODEL_LOADING_TIMEOUT: "3600"
        NUMBER_OF_GPU: 8
        INSTANCE_COUNT: 1
        HEALTH_CHECK_TIMEOUT: 300

    ```

Here, most of the configurations remain the same except:

- **Inference Script**: In the code sample above, since we are using the `ml.p4d.24xlarge` instance, we utilize the inference script [`p4d_hf_tgi.py`](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/scripts/p4d_hf_tgi.py). In this case we are using the HF model id we define in this configuration `meta-llama/Llama-2-70b-chat-hf` that is used in the script. 

- **image_uri**: The image uri will change based on the model you use. In this case, we are using the [HF TGI container image uri](https://aws.amazon.com/blogs/machine-learning/announcing-the-launch-of-new-hugging-face-llm-inference-containers-on-amazon-sagemaker/).

- **Model ENV variables**: Based on your model of choice, you can configure and change your model configurations that are used during deployment mentioned in the `config.yml file`.

Now, you should have the configurations for models, dataset, prompts and aws account details in this file, and we can load this file in all other notebooks to access this information for the steps below.

***View the [globals.py file](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/globals.py) which includes variables initialized to be loaded in different notebooks too.***

### Step 2: Set up the Environment and Deploy Model Endpoints

In this step, we will set up our test harness environment, download the requirements needed to run the environment, as well as deploy the model endpoints from the ***config.yml*** file asychronously.

1. Navigate to the file: [`0_setup.ipynb`](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/0_setup.ipynb): Run the notebook to import and download the requirements.txt that can be found here: [Requirements](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/requirements.txt). These requirements include:

    ```
    ipywidgets==8.1.1
    sagemaker==2.203.0
    transformers==4.36.2
    pandas==2.1.4
    datasets==2.16.1
    seaborn==0.13.1
    ```

2. Navigate to the file: [`1_deploy_model.ipynb`](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/1_deploy_model.ipynb): Run this notebook to deploy the models asychronously in different threads. The key components of this notebook are:

    - Loading the ***globals.py*** and ***config.yml*** file.

    - Setting a blocker function ***deploy_model*** to deploy the given model endpoint followed by:

    - A series of async functions to set tasks to deploy the models from the [config yml file](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/config-llama2-13b-inf2-g5-p4d-v1.yml) asynchronously in different threads. View the notebook from the link above.

    - Once the endpoints are deployed, their ***model configurations*** are stored within the [endpoints.json file](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/data/models/llama2-13b-inf2-g5-p4d-v1/endpoints.json).

At this point, you should have deployed the JumpStart/non JumpStart models and have your endpoints recorded in the respective ***endpoints.json*** file that can be found under the [models](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/tree/main/data/models) within the [data](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/tree/main/data) directory.


### Step 3: Generate data for model endpoint inference and benchmarking

In this step, we will run, download our dataset and generate prompts as payloads of different sizes to send to our different model endpoints with different combinations of concurrency levels for benchmarking metrics and visualizations.

1.  Navigate to the file: [`2_generate_data.ipynb`](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/2_generate_data.ipynb):

    This file will generate all data on `wikiqa (english version)` with prompt sizes `300 - 4000` token lengths in different payload sizes to send to the model endpoint during the inference pipeline. 
    
    You will also be able to generate the normal wikiqa dataset from the actual `long bench dataset`. The data can be downloaded from this link: [Dataset Link](https://github.com/THUDM/LongBench). This notebook has 3 main deliverables:

    - **Loading the dataset** that is stored within the dataset in the [data](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/tree/main/data) directory.

    - **Generating payloads**: Converts the loaded datasets into payloads based on the input question and records the context length of the prompt to send as a part of the payload during running inferences on the deployed endpoints. 
    
    This uses the [Prompt Template File](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/data/prompts/prompt_template.txt) to generate json files for the datasets that contain each element of the data (Question Answer Pair) as a ***Prompt payload*** that you can send to your model endpoints when we deploy them later in this repository. 
    
    This directory will then contain a jsonl file for each dataset as a list of payloads utilizing the prompt template of the model you use.

    All of the prompts are saved in this [data](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/tree/main/data/prompts) directory in a file named ***all_prompts.csv***.

    - **Constructing different sized payloads**: Laslty, this notebook divides the created prompts and payloads then filters it based on the sizes so the inferences can be sent in combination with concurrency levels. This way we can create combinations with first, payload `sizes of 1-500 token` length prompts, then `500-1000 token` length and so on.

At this point, you should have the payloads ready and segregated/filtered have it ready for inferencing the model with different combinations to test and benchmark for various metrics.

### Step 4: Run Inference on various combinations for benchmarking

In this step, we will run inferences on all deployed model endpoints ***(with a combination of different configurations, concurrency levels and payload sizes)***. We will call endpoints concurrently and asychronously to generate responses and record metrics. Here are some of the key components of the notebook:

1. Navigate to the file: [`3_run_inference.ipynb`](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/3_run_inference.ipynb): This notebook accomplishes the following:

    - **Accesses the deployed endpoints** that can be viewed [here](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/tree/main/data/models/llama2-13b-inf2-g5-p4d-v1) and creates a predictor object for these endpoints to call them during inference time.

    - **Functions to define metrics**: This notebook defines metrics to be recorded during the time of invocation for benchmarking purposes. An example for this function is `calculate_metrics`. Once these metrics are set, we start the process of creating inference using a series of functions.

    - **Running Actual Inferences**: Once the metrics are defined, we set a blocker function that is responsible for creating inference on a single payload called `get_inference`. We then run a series of asynchronous functions that can be viewed in the code (link above), to create asychronous inferefences on the deployed models. 
    
    - The way we send requests are by **creating combinations**: this means creating combinations of payloads of different sizes that can be viewed in the [config.yml file](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/config-llama2-13b-inf2-g5-p4d-v1.yml), with different concurrency levels (in this repository, concurrency level is set to 1, 2, and 4 but you can change it based on your requirements).

    - **Recording all metrics**: Once all inferences are completed, two main files are generated that can be viewed [here](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/tree/main/data/metrics/llama2-13b-inf2-g5-p4d-v1). These include the metrics recorded per inference, which includes metrics like `concurrency level` at which the specific model ran, the `completion`, `inference latency`, `average prompt token`, `total prompt token` and more. Another [all_metrics.csv file](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/data/metrics/llama2-13b-inf2-g5-p4d-v1/all_metrics.csv) is created with additional metrics.

Now that all metrics are recorded within the [metrics directory](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/tree/main/data/metrics), Run the next notebook to visually display these metrics to answer high value business questions like:

***I have a Llama2 70b, and I am running this on a 'p4d.24xlarge' instance. How many concurrent requests of sizes 500, 2000, and 3000 tokens can I send concurrently and successfully with the inference latency requirement of 9 seconds?***

With answers to questions like these, businesses and use cases can function more efficiently in terms of cost optimization, latency requirements, and time management.

### Step 5: View generated metrics

This step creates visualizations of bar charts and plots that answer compound questions targetting the best use of models for given `inference latency`, `transactions per second` and `cost` like metrics. Using visualizations from this notebook: you can focus not only on making sound technical decisions quicker, but focus on ***higher value business goals like performance efficiency and cost optimization***.

1. Navigate to the file: [`4_model_metric_analysis.ipynb`](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/4_model_metric_analysis.ipynb):

    This file contains certain portions: 

    - Utilizes existing metrics recorded from [here](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/tree/main/data/metrics) for generating charts.


    [CHART IMAGES - TO BE ADDED]
    [CHART IMAGES DESCRIPTION TO BE ADDED]

### Step 6: Clean Up

Once you have created your specific benchmarks and want to delete the endpoints that were created, navigate to the ['5_cleanup.ipynb'](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/5_cleanup.ipynb). This will delete all of the existing endpoints your created in service.

### Conclusion

In conclusion, this comprehensive benchmarking harness focuses on enabling customers and users to get: 

1. ***Flexibility of model choice to test with***: with various options of models and configurations, this harness serves as a stable template to use with no/low code changes and answer higher value business questions on cost and performance based on your use case.

2. ***Focus on valuable insights*** on metrics like `inference latency` in compliment with `concurrency levels` as well as `cost` and required `transactions per second`. We get visual represenations of how models function with different configurations and containers that makes it easy to make executive and technical decisions. 

3. With this smooth and low code technique of benchmarking models not only for small but large prompts of token sizes (3000-4000 tokens) becomes ***critical for several business use cases***. With the flexibility, you can add, change, remove and modify several configurations, metrics, and ways of model deployment in just a couple of steps. 

This gives a new level and potential to testing for all models out there, within or out of SageMaker JumpStart, for you to speed up your decision making processes and make it more efficient based on your key requirements.

