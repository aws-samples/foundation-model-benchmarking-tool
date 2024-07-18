Beyond running `FMBench` with the configuraton files provided, you may want try out bringing your own dataset or endpoint to `FMBench`. 

### Bring your own endpoint (a.k.a. support for external endpoints)

If you have an endpoint deployed on say `Amazon EKS` or `Amazon EC2` or have your models hosted on a fully-managed service such as `Amazon Bedrock`, you can still bring your endpoint to `FMBench` and run tests against your endpoint. To do this you need to do the following:

1. Create a derived class from [`FMBenchPredictor`](./src/fmbench/scripts/fmbench_predictor.py) abstract class and provide implementation for the constructor, the `get_predictions` method and the `endpoint_name` property. See [`SageMakerPredictor`](./src/fmbench/scripts/sagemaker_predictor.py) for an example. Save this file locally as say `my_custom_predictor.py`.

1. Upload your new Python file (`my_custom_predictor.py`) for your custom FMBench predictor to your `FMBench` read bucket and the scripts prefix specified in the `s3_read_data` section (`read_bucket` and `scripts_prefix`).

1. Edit the configuration file you are using for your `FMBench` for the following:
    - Skip the deployment step by setting the `2_deploy_model.ipynb` step under `run_steps` to `no`.
    - Set the `inference_script` under any experiment in the `experiments` section for which you want to use your new custom inference script to point to your new Python file (`my_custom_predictor.py`) that contains your custom predictor.

### Bring your own `REST Predictor` ([`data-on-eks`](https://github.com/awslabs/data-on-eks/tree/7173cd98c9be6f555afc42f8311cc7849f74a038) version)

`FMBench` now provides an example of bringing your own endpoint as a `REST Predictor` for benchmarking. View this [`script`](https://github.com/aws-samples/foundation-model-benchmarking-tool/blob/REST-predictor-fmbench/src/fmbench/scripts/rest_predictor.py) as an example. This script is an inference file for the `NousResearch/Llama-2-13b-chat-hf` model deployed on an [Amazon EKS](https://docs.aws.amazon.com/whitepapers/latest/overview-deployment-options/amazon-elastic-kubernetes-service.html) cluster using [Ray Serve](https://docs.ray.io/en/latest/ray-overview/examples.html). The model is deployed via `data-on-eks` which is a comprehensive resource for scaling your data and machine learning workloads on Amazon EKS and unlocking the power of Gen AI. Using `data-on-eks`, you can harness the capabilities of AWS Trainium, AWS Inferentia and NVIDIA GPUs to scale and optimize your Gen AI workloads and benchmark those models on FMBench with ease. 

### Bring your own dataset

By default `FMBench` uses the [`LongBench dataset`](https://github.com/THUDM/LongBench) dataset for testing the models, but this is not the only dataset you can test with. You may want to test with other datasets available on HuggingFace or use your own datasets for testing. You can do this by converting your dataset to the [`JSON lines`](https://jsonlines.org/) format. We provide a code sample for converting any HuggingFace dataset into JSON lines format and uploading it to the S3 bucket used by `FMBench` in the [`bring_your_own_dataset`](./src/fmbench/bring_your_own_dataset.ipynb) notebook. Follow the steps described in the notebook to bring your own dataset for testing with `FMBench`.

#### Support for Open-Orca dataset

Support for [Open-Orca](https://huggingface.co/datasets/Open-Orca/OpenOrca) dataset and corresponding prompts for Llama3, Llama2 and Mistral, see:

1. [bring_your_own_dataset.ipynb](https://github.com/aws-samples/foundation-model-benchmarking-tool/tree/main/src/fmbench/bring_your_own_dataset.ipynb)
1. [prompt templates](https://github.com/aws-samples/foundation-model-benchmarking-tool/tree/main/src/fmbench/prompt_template)
1. [Llama3 config file with OpenOrca](https://github.com/aws-samples/foundation-model-benchmarking-tool/tree/main/src/fmbench/configs/llama3/8b/config-llama3-8b-inf2-g5-byoe-w-openorca.yml)

### Building the `FMBench` Python package

If you would like to build a dev version of `FMBench` for your own development and testing purposes, the following steps describe how to do that.

1. Clone the `FMBench` repo from GitHub.

1. Make any code changes as needed.

1. Install [`poetry`](https://pypi.org/project/poetry/).
   
    ```{.bash}
    pip install poetry
    ```

1. Change directory to the `FMBench` repo directory and run poetry build.

    ```{.bash}
    poetry build
    ```

1. The `.whl` file is generated in the `dist` folder. Install the `.whl` in your current Python environment.

    ```{.bash}
    pip install dist/fmbench-X.Y.Z-py3-none-any.whl
    ```

1. Run `FMBench` as usual through the `FMBench` CLI command.