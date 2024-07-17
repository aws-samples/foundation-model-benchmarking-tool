
## Benchmark models deployed on different AWS Generative AI services

`FMBench` comes packaged with configuration files for benchmarking models on different AWS Generative AI services.

### Benchmark models on Bedrock

Choose any config file from the [`bedrock`](https://github.com/aws-samples/foundation-model-benchmarking-tool/tree/main/src/fmbench/configs/bedrock) folder and either run these directly or use them as templates for creating new config files specific to your use-case. Here is an example for benchmarking the `Llama3` models on Bedrock.

```{.bash}
fmbench --config-file https://raw.githubusercontent.com/aws-samples/foundation-model-benchmarking-tool/main/src/fmbench/configs/bedrock/config-bedrock-llama3.yml > fmbench.log 2>&1
```

### Benchmark models on SageMaker

Choose any config file from the model specific folders, for example the [`Llama3`](https://github.com/aws-samples/foundation-model-benchmarking-tool/tree/main/src/fmbench/configs/llama3) folder for `Llama3` family of models. These configuration files also include instructions for `FMBench` to first deploy the model on SageMaker using your configured instance type and inference parameters of choice and then run the benchmarking. Here is an example for benchmarking `Llama3-8b` model on an `ml.inf2.24xlarge` and `ml.g5.12xlarge` instance. 

```{.bash}
fmbench --config-file https://raw.githubusercontent.com/aws-samples/foundation-model-benchmarking-tool/main/src/fmbench/configs/llama3/8b/config-llama3-8b-inf2-g5.yml > fmbench.log 2>&1
```

### Benchmark models on EKS

You can use `FMBench` to benchmark models on hosted on EKS. This can be done in one of two ways:
 - Deploy the model on your EKS cluster independantly of `FMBench` and then benchmark it through the [Bring your own endpoint](#bring-your-own-endpoint-aka-support-for-external-endpoints) mode.
 - Deploy the model on your EKS cluster through `FMBench` and then benchmark it.
 
The steps for deploying the model on your EKS cluster are described below.

👉 **_EKS cluster creation itself is not a part of the `FMBench` functionality, the cluster needs to exist before you run the following steps_**. Steps for cluster creation are provided in [this](misc/eks_cluster-creation_steps.md) file but it would be best to consult the [DoEKS](https://github.com/awslabs/data-on-eks) repo on GitHub for comprehensive instructions.

1. Add the following IAM policies to your existing `FMBench` Role:

    1. [AmazonEKSClusterPolicy](https://docs.aws.amazon.com/aws-managed-policy/latest/reference/AmazonEKSClusterPolicy.html): This policy provides Kubernetes the permissions it requires to manage resources on your behalf.
    
    1. [AmazonEKS_CNI_Policy](https://docs.aws.amazon.com/aws-managed-policy/latest/reference/AmazonEKS_CNI_Policy.html): This policy provides the Amazon VPC CNI Plugin (amazon-vpc-cni-k8s) the permissions it requires to modify the IP address configuration on your EKS worker nodes. This permission set allows the CNI to list, describe, and modify Elastic Network Interfaces on your behalf.
    
    1. [AmazonEKSWorkerNodePolicy](https://docs.aws.amazon.com/aws-managed-policy/latest/reference/AmazonEKSWorkerNodePolicy.html): This policy allows Amazon EKS worker nodes to connect to Amazon EKS Clusters.
 
1. Once the EKS cluster is available you can use either the following two files or create your own config files using these files as examples for running benchmarking for these models. **_These config files require that the EKS cluster has been created as per the steps in these [instructions](https://awslabs.github.io/data-on-eks/docs/gen-ai/inference/llama3-inf2)_**.

    1. [config-llama3-8b-eks-inf2.yml](https://github.com/aws-samples/foundation-model-benchmarking-tool/blob/main/src/fmbench/configs/llama3/8b/config-llama3-8b-eks-inf2.yml): Deploy Llama3 on Trn1/Inf2 instances.
    
    2. [config-mistral-7b-eks-inf2.yml](https://github.com/aws-samples/foundation-model-benchmarking-tool/blob/main/src/fmbench/configs/mistral/config-mistral-7b-eks-inf2.yml): Deploy Mistral 7b on Trn1/Inf2 instances.
    
    For more information about the [blueprints](https://github.com/aws-samples/foundation-model-benchmarking-tool/tree/main/src/fmbench/configs/eks_manifests) used by FMBench to deploy these models, view: [DoEKS docs gen-ai](https://awslabs.github.io/data-on-eks/docs/gen-ai).
    
1. Run the `Llama3-8b` benchmarking using the command below (replace the config file as needed for a different model). This will first deploy the model on your EKS cluster and then run benchmarking on the deployed model.

    ```{.bash}
    fmbench --config-file https://raw.githubusercontent.com/aws-samples/foundation-model-benchmarking-tool/main/src/fmbench/configs/llama3/8b/config-llama3-8b-eks-inf2.yml > fmbench.log 2>&1
    ```

1. As the model is getting deployed you might want to run the following `kubectl` commands to monitor the deployment progress. Set the _model_namespace_ to `llama3` or `mistral` or a different model as appropriate.

    1. `kubectl get pods -n <model_namespace> -w`: Watch the pods in the model specific namespace.
    1. `kubectl -n karpenter get pods`: Get the pods in the karpenter namespace.
    1. `kubectl describe pod -n <model_namespace> <pod-name>`: Describe a specific pod in the mistral namespace to view the live logs.

### Benchmark models on EC2

You can use `FMBench` to benchmark models on hosted on EC2. This can be done in one of two ways:
 - Deploy the model on your EC2 instance independantly of `FMBench` and then benchmark it through the [Bring your own endpoint](#bring-your-own-endpoint-aka-support-for-external-endpoints) mode.
 - Deploy the model on your EC2 instance through `FMBench` and then benchmark it.
 
The steps for deploying the model on your EC2 instance are described below. 

👉 In this configuration both the model being benchmarked and `FMBench` are deployed on the same EC2 instance.

1. Create a new EC2 instance suitable for hosting an LMI as per the steps described [here](misc/ec2_instance_creation_steps.md).

1. Install `FMBench` on this instance and run benchmarking for a desired model using one of the config files included in the `FMbench` repo or create your own.

    1. Connect to your instance using any of the options in EC2 (SSH/EC2 Connect), run the following in the EC2 terminal. This command installs Anaconda on the instance which is then used to create a new `conda` environment for `FMBench`. See instructions for downloading anaconda [here](https://www.anaconda.com/download)
    
        ```{.bash}
        curl -O https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
        chmod +x Anaconda3-2023.09-0-Linux-x86_64.sh
        ./Anaconda3-2023.09-0-Linux-x86_64.sh
        export PATH=/home/ubuntu/anaconda3/bin:$PATH
        ```

    1. Setup the `fmbench_python311` conda environment.

        ```{.bash}
        conda create --name fmbench_python311 -y python=3.11 ipykernel
        source activate fmbench_python311;
        pip install -U fmbench
        ```

    1. Create local directory structure needed for `FMBench` and copy all publicly available dependencies from the AWS S3 bucket for `FMBench`. This is done by running the `copy_s3_content.sh` script available as part of the `FMBench` repo.

        ```{.bash}
        curl -s https://raw.githubusercontent.com/aws-samples/foundation-model-benchmarking-tool/main/copy_s3_content.sh | sh
        ```

    1. To download the model files from HuggingFace, create a `hf_token.txt` file in the `/tmp/fmbench-read/scripts/` directory containing the Hugging Face token you would like to use. In the command below replace the `hf_yourtokenstring` with your hugging Face token.
    
        ```{.bash}
        echo hf_yourtokenstring > /tmp/fmbench-read/scripts/hf_token.txt
        ```
    
    1. Run `FMBench` with a packaged or a custom config file. **_This step will also deploy the model on the EC2 instance_**. The `--write-bucket` parameter value is just a placeholder and an actual S3 bucket is not required

        ```{.bash}
        fmbench --config-file /tmp/fmbench-read/configs/llama3/8b/config-ec2-llama3-8b.yml --local-mode yes --write-bucket placeholder > fmbench.log 2>&1
        ```
    
    1. For example, to run `FMBench` on a `llama3-8b-Instruct` model on an `inf2.48xlarge` instance, run the command 
    command below. The config file for this example can be viewed [here](src/fmbench/configs/llama3/8b/config-ec2-llama3-8b-inf2-48xl.yml).

        ```{.bash}
        fmbench --config-file /tmp/fmbench-read/configs/llama3/8b/config-ec2-llama3-8b-inf2-48xl.yml --local-mode yes --write-bucket placeholder > fmbench.log 2>&1
        ```

    1. Open a new Terminal and navigate to the `foundation-model-benchmarking-tool` directory and do a `tail` on `fmbench.log` to see a live log of the run.

        ```{.bash}
        tail -f fmbench.log
        ```

    1. All metrics are stored in the `/tmp/fmbench-write` directory created automatically by the `fmbench` package. Once the run completes all files are copied locally in a `results-*` folder as usual.