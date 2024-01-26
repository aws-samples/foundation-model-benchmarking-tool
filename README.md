# Foundation Model benchmarking tool (FMBT) built using Amazon SageMaker

A key challenge with FMs is the ability to benchmark their performance in terms of inference latency, throughput and cost so as to determine which model running with what combination of the hardware and serving stack provides the best price-performance combination for a given workload.

Stated as **business problem**, the ask is “_*What is the dollar cost per transaction for a given generative AI workload that serves a given number of customers while keeping the response time under a target threshold?*_”

But to really answer this question, we need to answer an **engineering question** (an optimization problem, actually) corresponding to this business problem: “*_What is the minimum number of instances N, of most cost optimal instance type T, that are needed to serve a workload W while keeping the average transaction latency under L seconds?_*”

*W: = {R transactions per-minute, average prompt token length P, average generation token length G}*

This foundation model benchmarking tool (a.k.a. FMBT) is a tool to answer the above engineering question.

## Functionality

The FMBT provides the following capabilities.

1. Create a dataset of different prompt sizes and select one or more such datasets for running the tests.
    1. Currently use datasets from [LongBench](https://github.com/THUDM/LongBench) and filter out individual items from the dataset based on their size in tokens (for example, prompts less than 500 tokens, between 500 to 1000 tokens and so on and so forth).
2. Deploy **any model** that is deployable on SageMaker on **any supported instance type** (`g5`, `p4d`, `Inf2`).
    1. Models could be either available via SageMaker JumpStart (list available [here](https://sagemaker.readthedocs.io/en/stable/doc_utils/pretrainedmodels.html)) as well as models not available via JumpStart but still deployable on SageMaker through the low level boto3 (Python) SDK (Bring Your  Own Script).
    2. Model deployment is completely configurable in terms of the inference container to use, environment variable to set, `setting.properties` file to provide (for inference containers such as DJL that use it) and instance type to use.
3. Benchmark FM performance in terms of inference latency, transactions per minute and dollar cost per transaction for any FM that can be deployed on SageMaker.
    1. Tests are run for each combination of the configured concurrency levels i.e. transactions (inference requests) sent to the endpoint in parallel and dataset. For example, run multiple datasets of say prompt sizes between 3000 to 4000 tokens at concurrency levels of 1, 2, 4, 6, 8 etc. so as to test how many transactions of what token length can the endpoint handle while still maintaining an acceptable level of inference latency.
4. Generate a report comparing and contrasting the performance of the models over different test configurations.
    1. The report is generated in the [Markdown](https://en.wikipedia.org/wiki/Markdown) format and consists of plots, tables and text that highlight the key results and provide an overall recommendation on what is the best combination of instance type and serving stack to use for the model under stack for a dataset of interest. 
    2. The report is created as an artifact of reproducible research so that anyone having access to the model and the serving stack (inference container, instance type) can run the code and recreate the same results and report.
5. The entire FMBT code base is available on [this GitHub repo](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness) and contains different [configuration files](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/tree/main/configs) that can be used as reference for benchmarking new models and instance types.

## How it works

The code is a collection of Jupyter Notebooks that are run in order to benchmark a desired model on a desired set of instance types.

1. The FMBT is currently intended to run on SageMaker (or any other compute platform where Python 3.11 and JupyterLab can be installed).
    1. While the solution can technically run anywhere (including on a non-AWS environment for development and testing) but we do want to run it on AWS compute in order to avoid counting internet round trip time as part of the model latency.
2. Clone the [FMBT code repo](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness.git) (you would likely want to fork the repo to create your own copy).
3. Create a config file in the [configs](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/tree/main/configs) directory.
    1. The configuration file is a YAML file containing configuration for all steps of the benchmarking process. It is recommended to create a copy of an existing config file and tweak it as necessary to create a new one for your experiment.
    2. Change the config file name in the [config_filename.txt](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/config_filepath.txt) to point to your config file.
4. Run the [setup notebook](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/0_setup.ipynb) to install the required [Python packages](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/requirements.txt).
5. Setup the Llama tokenizer and datasets needed for download as per instructions in this [README](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/tree/main?tab=readme-ov-file#solution-prerequisites).
6. Run the [dataset preparation notebook](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/2_generate_data.ipynb) to create the prompt payloads ready for testing.
7. Run the [model deployment notebook](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/1_deploy_model.ipynb) to deploy models on different endpoints with the desired configuration as per the configuration file.
    1. If you are using a model not supported through JumpStart than you can place your deployment script in the [scripts](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/tree/main/scripts)directory and set the deployment script name in the configuration file. Your deployment script needs to have a `deploy_model` that the FMBT code will call to deploy the model (refer to existing scripts in the scripts director for reference).
8. Run the [inference notebook](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/3_run_inference.ipynb) to run inference on the deployed endpoints and collect metrics. These metrics are saved in the metrics directory (these raw metrics are not checked in back into the repo).
9. Run the [report generation notebook](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/4_model_metric_analysis.ipynb) to create statistical summaries, plots, tables and a [final report](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/data/metrics/llama2-13b-inf2-g5-p4d-v1/results.md) for the test results.
10. Run the [cleanup notebook](https://github.com/aws-samples/jumpstart-models-benchmarking-test-harness/blob/main/5_cleanup.ipynb) to delete the deployed endpoints.
11. Commit and push your code (this will include the results) into the GitHub repo once you are done with your experiment.

## Results

Here is a screenshot of the `report.md` file generated b FMBT.
![Report](./img/results.gif)

## Pending enhancements

The following enhancements are identified as next steps for FMBT.

1. [**Highest priority**] Containerize FMBT and provide support for running this both as a SageMaker Processing Job as well as on AWS Fargate. This would also include shifting all data and result storage to Amazon S3.
2. Support for a custom token counter. Currently only the LLama tokenizer is supported but we want to allow users to bring their own token counting logic for different models.
3. Support for different payload formats that might be needed for different inference containers. Currently the HF TGI container, and DJL Deep Speed container on SageMaker both use the same format but in future other containers might need a different payload format.
4. Emit live metrics so that they can be monitored through Grafana via live dashboard.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](./LICENSE) file.
