# Benchmarking for Amazon SageMaker JumpStart models and bring your own models


*Authors*: *Amit Arora*, *Madhur Prashant*

[Amazon SageMaker JumpStart](https://aws.amazon.com/sagemaker/jumpstart/getting-started/) offers state-of-the-art foundation models for use cases such as content writing, code generation, question answering, copywriting, summarization, classification, information retrieval, and more. A key challenge with FMs is the ability to benchmark their performance in terms of inference latency, throughput and cost so as to determine which model running with what combination of the following is the most optimal option for a given use-case:

- Inference container and parameters (for example: batch size, degree of tensor parallelism etc.)
- Runtime inference parameters (for example temperature, top_p etc.)
- Choice of hardware (for example: `p4d`, `Inf2`, `g5` etc.)

## Solution overview

The solution presented in this repository provides a test harness for doing this benchmarking for not only Sagemaker Jumpstart FMs, but models out of jumpstart where you can bring your own inference script and servings file, and run this repo as a test for inference latency, token throughput, concurrency metrics, and more. In this solution, we are using the LLaMa models (13b and 70b), and details on these models can be viewed [here](https://ai.meta.com/llama/). For both jumpstart and non jumpstart models, we are using the instance types below, to test for metrics including inference latency, throughput and concurrency metrics (how many requests can a single endpoint with instance_type 'x' handle with a requirement of 'y' latency?).

***The instances that we are using in this solution, (but is not limited to these instance types) are:***

- **p4d based instances**: p4d.24xlarge
- **g5 based instances**: g5.12xlarge, g5.24xlarge, g5.48xlarge
- **Inferentia based instances**: inf2.24xlarge, inf2.48xlarge

For bring your own model, we are supporting the latest versions of the DJL image uri that can be viewed in this [DJL documentation](https://docs.djl.ai/docs/serving/serving/docs/large_model_inference.html).


## Solution Pre Requisites:

To run this open source benchmarking harness for your specific use case, follow the pre requisites below to set up your environment before running the content and generating tests:

- *Llama 2 Tokenizer Requirements* : The use of this model is governed by the Meta license. In order to download the model weights and tokenizer, please visit the website and accept our License before requesting access here: [meta approval form](https://ai.meta.com/resources/models-and-libraries/llama-downloads/). Once you have been approved, please download the following files into the [llama2_tokenizer](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/tree/main/llama2_tokenizer) directory: ***tokenizer_config.json, tokenizer.model, tokenizer.json, special_tokens_map.json, pytorch_model.bin.index.json, model.safetensors.index.json, generation_config.json, config.json***

To access the Llama 2 files after being approved, view the [Hugging Face website](https://huggingface.co/meta-llama/Llama-2-7b/tree/main) for this model to download the files mentioned above.

- *Data Ingestion* : In this repository example, we used prompts from the LongBench dataset that can be viewed [here](https://github.com/THUDM/LongBench). To replicate the process, please download the different files into this [data](https://github.com/THUDM/LongBench) directory within a folder that you create named 'dataset'. Here are the file names that you can download from this website to get started:

    - 2wikimqa
    - hotpotqa
    - narrativeqa
    - triviaqa

    Once you have uploaded these files within your dataset folder in your data directory, use the [Prompt Template File](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/data/prompts/prompt_template.txt) to generate json files for the above datasets that contain each element of the data (Question Answer Pair) as a ***Prompt payload*** that you can send to your model endpoints when we deploy them later in this repository. This directory will then contain a jsonl file for each dataset as a list of payloads utilizing the prompt template of the model you use.

- *Hugging Face Auth Tokens For file access* : For specific models, a requirement would be to download files from ***Hugging Face*** utilizing your Auth Token from this code repository. To define your auth token, create a ***hf_token.txt*** file containing your auth token. Once this is done, you will be able to download files without any errors. 

For privacy concerns, the *hf_token.txt* you define will be checked into the [.gitignore file](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/.gitignore), so that your auth token is not exposed when you check your code into repositories online.

## Solution Design:

The solution design consists of 6 steps. This includes setting up your aws environment, configuring a [yml file](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/config.yml) that will contain information that will be loaded into all executable notebooks in our benchmarking process. We will then focus on accessing different Sagemaker jumpstart and non jumpstart model and deploying them seamlessly via this solution in your aws account. Once we have these endpoints deployed, we will then generate all data and create payloads within your [data](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/tree/main/data) directory as mentioned in the prerequisites above. To make this benchmarking smoother and more granular in terms of the context token length that each model can take in, we will create payloads of different sizes for example, ***1-500 tokens, 500-1000 tokens, 1000-2000 tokens, 2000-3000 tokens and finally 3000-4000 tokens***. After generating data, we will run inference using [this notebook](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/3_run_inference.ipynb) concurrently and asynchronously on each deployed endpoint with different combination of payload sizes and concurrency metrics, and track the inference latency and other metrics we will display via the [metrics analysis](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/4_model_metric_analysis.ipynb) notebook. 


### Step 1: Configure your Config.yml file

Navigate to [config.yaml file](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/config.yml), Configure your AWS account in your IDE environment, using <code>aws configure</code>, enter your access and secret access keys. Enter your Amazon SageMaker execution role and region here

    ```
    aws:
      region: <region-name>
      sagemaker_execution_role: <execution-role-name>
    ```
      Make sure your execution role has the right permissions.

This ***config.yml*** file consists of:

- Information on the sizes of your dataset payloads.
- Inference Parameters: This is based on your model and requirements. In this example, we are using different ***Llama 2 Models***
- ***Model Configurations (experiments)*** : This portion of the file contains the model configurations for the sagemaker jumpstart and non jumpstart models that you want to deploy. These are the configurations that will be referred to for model configurations to use to deploy the samgemaker jumpstart model. This configuration consists of two main parts: configurations for ***sagemaker jumpstart models*** and ***bring your own model***. Here are two examples:

    1. Configurations for the Sagemaker Jumpstart Model: ***meta-textgeneration-llama-2-13b***:
    # ============================

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

    Here, we are configuring the version of the model, the endpoint name, model_id that needs to be deployed. Configurations also support the gives instance type to be used, in this case: "ml.g5.24xlarge", the image uri, whether or not to deploy this given model, followed by an inference script ["jumpstart.py"](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/scripts/jumpstart.py) which supports the inference script for jumpstart models to deploy the model in our main [deploy notebook](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/1_deploy_model.ipynb). Then you can set the payload files of different prompt sizes as above (for 1-500 tokens, 500-1000 tokens and so on until 4000 token length prompts). Next, you can ***set the concurrency levels, so the payloads will be combined with each concurrency level and the benchmarking will take place accordingly***. Lastly, set some final environment variables to use during model deployment such as the SAGEMAKER_PROGRAM, ENDPOINT_SERVER_TIMEOUT, and the rest above.

    2. Configurations for the Bring Your Own Model: ******:
    # ============================

        ```
         # P4D Based Instance Model Configuration:
        - name: llama2-70b-p4d.24xlarge-tgi-inference-2.0.1-tgi0.9.3-gpu-py39-cu118
            model_id: meta-llama/Llama-2-70b-hf
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

    - ***Inference Script***: Here, we are using the inference script [p4d_hf_tgi.py](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/scripts/p4d_hf_tgi.py). In this case we are using the HF model id we define in this configuration `meta-llama/Llama-2-70b-hf` that is used in the script. 

    - ***image_uri***: The image uri will change based on the model you use. In this case, we are using the HF TGI container image uri.

    - ***Model ENV variables***: Based on your model of choice, you can configure and change your model configurations here that are used respectively in the config.yml file.


So now that we have the configurations for models, the dataset, prompts and aws account details in this file, we can load this file in all other notebooks and access this information. 

***View the [globals.py file](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/globals.py) which includes variables initialized to be loaded in different notebooks too.***

### Step 2: Set up the environment and Deploy Model Endpoints

This step of our solution design covers setting up the environment, downloading the requirements needed to run the environment, as well as deploying the model endpoints from the ***config.yml*** file asychronously.

1. Navigate to the file: [`0_setup.ipynb`](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/0_setup.ipynb): Run the cell to import and download the requirements.txt that can be found here: [Requirements](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/requirements.txt)

2. Navigate to the file: [`1_deploy_model.ipynb`](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/1_deploy_model.ipynb): Run this notebook to deploy the models asychronously in different threads. The key components of this notebook for the purposes of understanding are:

    - Loading the ***globals.py*** and ***config.yml*** file.
    - Setting a blocker function ***deploy_model*** to deploy the given model endpoint followed by:
    - A series of async functions to set tasks to deploy the models from the [config yml file](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/config-llama2-13b-inf2-g5-p4d-v1.yml) asynchronously in different threads. View the notebook from the link above.
    - Once the endpoints are deployed, their ***model configurations*** are stored within the [endpoints.json file](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/data/models/llama2-13b-inf2-g5-p4d-v1/endpoints.json).

At this point, you should have deployed your jumpstart/non jumpstart model and have had your endpoints recorded in the respective ***endpoints.json*** file that can be found under the [models](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/tree/main/data/models) within the [data](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/tree/main/data) directory.


### Step 3: Generate data for model endpoint inference and benchmarking

This step of our solution design includes running and downloading our dataset, generating prompts as payloads of different sizes that we will send to our different model endpoints with different combinations of concurrency levels that we will later use to run inference and generate benchmarking metrics and visualizations.

1.  Navigate to the file: [`2_generate_data.ipynb`](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/2_generate_data.ipynb):

    This file will generate all data on wikiqa (english version) with prompt sizes 300 - 4000 token lengths in different payload sizes to send to the model endpoint during the inference pipeline. You will also be able to generate the normal wikiqa dataset from the actual 'long bench dataset'. The data can be downloaded from this link: [Dataset Link](https://github.com/THUDM/LongBench). This notebook then focuses on 3 main deliverables:

    - ***Loading the dataset*** that is stored within the dataset in the [data](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/tree/main/data) directory.

    - ***Generating payloads***: This notebook also converts the loaded datasets into payloads based on the input question and records teh context length of the prompt to send as a part of the payload during running inferences on the deployed endpoints.

    All of the prompts are saved in this [data](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/tree/main/data/prompts) directory in a file named ***all_prompts.csv***.

    - ***Constructing different sized payloads***: This notebook lastly divides the created prompts and payloads and filters it based on the sizes so the inferences can be sent in combination with concurrency levels in a way such that we create combinations with first, payload sizes of 1-500 token length prompts, then 500-1000 token length and so on.

At this point, you should have the payloads ready and segregated/filtered based on the ***prompt truncate level*** and now have it ready for inferencing the model with different combinations to test and benchmark for various metrics.


### Step 4: Run Inference on various combinations for benchmarking

This step of our solution design includes running inferences on all deployed model endpoints (with different configurations, concurrency levels and payload sizes). This notebook runs inferences in a manner that is calls endpoints concurrently and asychronously to generate responses and record metrics. Here are some of the key components of the notebook to understand:

1. Navigate to the file: [`3_run_inference.ipynb`](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/3_run_inference.ipynb):

    As you view the contents of the notebook, there are a couple tasks occuring within it as seen below:

    - Accessing the deployed endpoints that can be viewed [here](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/tree/main/data/models/llama2-13b-inf2-g5-p4d-v1), creating a predictor object for these endpoints to call them during inference time.

    - ***Functions to define metrics***: This notebook sets stage for metrics to be recorded during the time of invocation of all these models for benchmarking purposes. An example for this function is `calculate_metrics`. Once these metrics are set, we start the process of creating inference using a series of functions.

    - ***Running Actual Inferences***: Once the metrics are defined, we set a blocker function that is responsible for creating inference on a single payload called `get_inference`. We then run a series of asynchronous functions that can be viewed in the code (link above), to create asychronous inferefences on the deployed models. The way we send requests are by ***creating combinations***: this means creating combinations of payloads of different sizes that can be viewed in the [config.yml file](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/config-llama2-13b-inf2-g5-p4d-v1.yml), with different concurrency levels (in this case we first go through all patches of payloads with a concurrency level of 1, then 2, and then 4). You can set this to your desired value.

    - ***Recording all metrics***: Once all inferences are completed, two main files are generated that can be viewed [here](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/tree/main/data/metrics/llama2-13b-inf2-g5-p4d-v1). This includes the metrics recorded per inference, which includes metrics like ***concurrency level*** at which the specific model ran, the ***completion, inference latency, average prompt token, total prompt token*** and more. Another [all_metrics.csv file](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/data/metrics/llama2-13b-inf2-g5-p4d-v1/all_metrics.csv) is created with additional metrics.

Now that we have recorded the metrics within the [metrics directory](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/tree/main/data/metrics), we can go ahead and visually display these metrics to answer questions like:

***I have a Llama2 70b, and I am running this on a 'p4d.24xlarge' instance. How many concurrent requests of sizes 500, 2000, and 3000 tokens can I send concurrently and successfully with the inference latency requirement of 9 seconds?***

### Step 5: View generated metrics

1. Navigate to the file: [`4_model_metric_analysis.ipynb`](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/4_model_metric_analysis.ipynb):

    This file contains certain portions: 

    - Utilizes existing metrics recorded from [here](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/tree/main/data/metrics) for generating charts.


    [CHART IMAGES - TO BE ADDED]
    [CHART IMAGES DESCRIPTION TO BE ADDED]

### Step 6: Clean Up

This is the final portion of the solution design. Once you have created your specific benchmarks and want to delete the endpoints that were created, navigate to the ['5_cleanup.ipynb'](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/5_cleanup.ipynb). This will delete all of the existing endpoints your created in service.

### Conclusion

This benchmark focuses on enabling customers and users to get 1/flexibility of model choice to test with, 2/valuable business insights on metrics like inference latency in compliment with concurrency levels and visual represenations of how models function with different configurations on different containers and more. With this smooth and low code technique of benchmarking models not only for small but large prompts of token sizes (3000-4000 tokens) becomes critical for several business use cases. With the flexibility, you can add, change, remove and modify several configurations, metrics, and ways of model deployment in just a couple of steps. This gives a new level and potential to testing for all models out there, within or out of sagemaker jumsptart, for you to speed up your decision making processes and make it more efficient based on your key requirements.
