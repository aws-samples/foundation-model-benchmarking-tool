# Benchmarking for Amazon SageMaker JumpStart models

[Amazon SageMaker JumpStart](https://aws.amazon.com/sagemaker/jumpstart/getting-started/) offers state-of-the-art foundation models for use cases such as content writing, code generation, question answering, copywriting, summarization, classification, information retrieval, and more. A key challenge with FMs is the ability to benchmark their performance in terms of inference latency, throughput and cost so as to determine which model running with what combination of the following is the most optimal option for a given use-case:
- Inference container and parameters (for example: batch size, degree of tensor parallelism etc.)
- Runtime inference parameters (for example temperature, top_p etc.)
- Choice of hardware (for example: `p4d`, `Inf2` etc.)

This repository provides a test harness for doing this benchmarking (note that this is different from benchmarking accuracy of a particular model for a given use-case).

## Instructions:

1. Configure your AWS account in your IDE environment, using 'aws configure', enter your access and secret access keys.

1. Navigate to [`config.yaml`](./config.yml), enter your Amazon SageMaker execution role and region here:

    ```
    aws:
      region: <region-name>
      sagemaker_execution_role: <execution-role-name>
    ```
      Make sure your execution role has the right permissions.

1. Navigate to the file: `0_llama_deploy_setup.ipynb` - this is where we are deploying the models, where you have the ability to configure the model parameters and configurations. Model options are: LLaMa-2-7b, 13b, and chat models that are supported by Neuron now on Sagemaker Jumpstart. 

    Supported instances include all Inferentia 2 instances: inf2.xlarge, inf2.8xlarge, inf2.24xlarge, inf2.48xlarge

    Incoming instances to be testing for llama-2 models: p4d endpoint instance (TBD - still to experiment with this - requested quota on the account)

    Run the entire notebook (0_llama_deploy) and wait for the endpoints to be deployed (this can be viewed on the console after running the cell, or you can double check on the aws console)

4. Once all models are deployed, to make sure, information on models that are deployed, and their configurations can be viewed in the following file: Model_details/global_endpoints.json as follows:

    ```
    {
        "meta-textgenerationneuron-llama-2-7b": {
            "_model_data_is_set": false,
            "orig_predictor_cls": null,
            "model_id": "meta-textgenerationneuron-llama-2-7b",
            "model_version": "*",
            "instance_type": "ml.inf2.24xlarge",
            "resources": "ResourceRequirements: {'requests': {'num_accelerators': 1, 'memory': 8192}, 'limits': {}, 'num_accelerators': 1, 'min_memory': 8192, 'copy_count': 1}",
            "tolerate_vulnerable_model": false,
            "tolerate_deprecated_model": false,
            "region": "us-east-1",
            "sagemaker_session": "<sagemaker.session.Session object at 0x1209e7b10>",
            "model_data": {
                "S3DataSource": {
                    "S3Uri": ,
                    "S3DataType": "S3Prefix",
                    "CompressionType": "None",
                    "ModelAccessConfig": {
                        "AcceptEula": true
                    }
                }
            },
            "image_uri": "763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.24.0-neuronx-sdk2.14.1",
            "predictor_cls": "<class 'sagemaker.base_predictor.Predictor'>",
            "name": "",
            "_base_name": null,
            "algorithm_arn": null,
            "model_package_arn": null,
            "_sagemaker_config": null,
            "role": "AmazonSageMaker-ExecutionRole-20230807T175994",
            "vpc_config": null,
            "endpoint_name": "",
            "_is_compiled_model": false,
            "_compilation_job_name": null,
            "_is_edge_packaged_model": false,
            "inference_recommender_job_results": null,
            "inference_recommendations": null,
            "_enable_network_isolation": true,
            "env": {
                "OPTION_DTYPE": "fp16",
                "OPTION_N_POSITIONS": "4096",
                "OPTION_TENSOR_PARALLEL_DEGREE": "4",
                "OPTION_MAX_ROLLING_BATCH_SIZE": "4",
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
                "OPTION_ROLLING_BATCH": "auto",
                "OPTION_NEURON_OPTIMIZE_LEVEL": "2"
            },
            "model_kms_key": null,
            "image_config": null,
            "entry_point": null,
            "source_dir": null,
            "dependencies": [],
            "git_config": null,
            "container_log_level": 20,
            "bucket": null,
            "key_prefix": null,
            "uploaded_code": null,
            "repacked_model_data": null,
            "mode": null,
            "modes": {},
            "serve_settings": null,
            "content_types": null,
            "response_types": null,
            "accept_eula": true
        }
    ```


    This file will be used as a global file for instance management and will be updated as more endpoints are deployed and will only contain the endpoints that are actively in service. Now, move to the next notebook:

1.  Now run the notebook: `1_generate_data.ipynb`

    This file will generate all data on wikiqa (english version) with prompt sizes 300 - 4000 token lengths. You will also be able to generate the normal wikiqa dataset from the actual 'long bench dataset'. The data can be viewed as follows once you run this notebook:

    - `data/prompts`: this contains new prompts (that are prompts attained from the filtered dataset) and 'json_requests' that are prompts from the unfiltered 'wikiqa' dataset within long bench.

    - `2wikimqa_e.jsonl` is a file within 'data' that contains the unfiltered prompts - we are only using the filtered prompts in this dataset for inference from neuron based llama models deployed on large inf2 instances.

1.  Download the llama2_tokenizer from the hugging face website and fill out the Meta approval form to get access to llama models and tokenizer files. Otherwise, use the files that are in this repo for faster tokenization of prompts in datasets.

    After running this file, you will have unfiltered prompts saved as json_requests to data/prompts/json_requests.json. The filtered prompts are pre configered as new_prompts.json within the data folder. Fetch that from this repo as is.

    Now, let's load this data and generate inferences from our deployed model endpoints:

1.  Now run the notebook: 2_load_data_inference.ipynb. While you run this notebook, there are certain tasks being handled: 

1. Grab global instances from the global json file with all of the endpoint configuration information. All active endpoints are retrieved as predictors for the models to perform prediction on, and the inactive endpoints are deleted from the file automatically. Here is an example of the endpoints that are active and returned for prediction purposes:

    ```
    Endpoint: 
      model_id: meta-textgenerationneuron-llama-2-7b
      instance_type: ml.inf2.24xlarge
      batch_size: 4
      n_positions: 4096
      tensor_parallel_degree: 4

    Endpoint: 
      model_id: meta-textgenerationneuron-llama-2-7b_3411
      instance_type: ml.inf2.48xlarge
      batch_size: 4
      n_positions: 4096
      tensor_parallel_degree: 4
    ```
    Now, once we have the active endpoints, we run the code below:

1.  !python3 /Users/madhurpt/llama-2-neuron-benchmarking/async_inference.py
    this executes a python script that handles concurrent asynchronous calling of each of the endpoint. You can configure this file and experiment with the number of concurrent callings using semaphores on each endpoint that is deployed. 

    You will be able to view the process of prompt predictions for each endpoint once you execute the cell - there are 105 filtered prompts in the new_prompts.json file, so if you have two endpoints as deployed in this notebook, you will have 210 predictions (one per endpoint).

    Executable time (when concurrent requests per endpoint is 3-4 (when there are two llama-2-7b endpoints)) = 65 minutes.

    Once prompt predictions are completed, view the: Inference.csv file that contains: inf. latency, transactions per second, token throughput and more metrics per endpoint.



1.  Now run the notebook: 3_model_metrics file.ipynb. 

    This is flexible to change but returns the number of prompts that threw an error and the number that did not, their average token count input length and latency. Observations for 2 llama-2-7bs deployed on inf2.24x and inf2.48x large:

    ```
    **Metric**	                                                    **Error Prompts**	    **No Error Prompts**
    **Over 2000 and Under 3000 Tokens **                            	1.0	                   61.0
    **Over 3000 Tokens**	                                            69.0	                 45.0
    **Average Input Token Length**	                              3656.2285714285700	
    **Average Input Token Length**		                                                   2473.3714285714300
    **Average Output Token Length**		                                                    48.23571428571430
    **Largest Token Length**		                                                             3420.0
    **Smallest Token Length	**	                                                              2.0
    ```


