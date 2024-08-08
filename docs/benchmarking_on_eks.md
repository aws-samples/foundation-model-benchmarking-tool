# Benchmark models on EKS

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
