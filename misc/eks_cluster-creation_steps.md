# EKS cluster creation steps

The steps below create an EKS cluster called `trainium-inferentia`.

1. Before we begin, ensure you have all the prerequisites in place to make the deployment process smooth and hassle-free. Ensure that you have installed the following tools on your machine: [aws-cli](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html), [kubectl](https://docs.aws.amazon.com/eks/latest/userguide/install-kubectl.html) and [terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli). We use the [`DoEKS`](https://github.com/awslabs/data-on-eks/tree/main) repository as a guide to deploy the cluster infrastructure in an AWS account.


1. Clone the [`DoEKS`](https://github.com/awslabs/data-on-eks) repository

    ``` {.bash}
    git clone https://github.com/awslabs/data-on-eks.git
    ```

1. Ensure that the region names are correct in [`variables.tf`](https://github.com/awslabs/data-on-eks/blob/d532720d0746959daa6d3a3f5925fc8be114ccc4/ai-ml/trainium-inferentia/variables.tf#L12) file before running the cluster creation script.

1. Ensure that the ELB to be created would be external facing. Change the helm value from `internal` to `internet-facing` [here](https://github.com/awslabs/data-on-eks/blob/3ef55e21cf30b54341bb771a2bb2dbd1280c3edd/ai-ml/trainium-inferentia/helm-values/ingress-nginx-values.yaml#L8).

1. Ensure that the IAM role you are using has the permissions needed to create the cluster.

    1. Attach the following managed policies: `AmazonEKSClusterPolicy`, `AmazonEKS_CNI_Policy`, and `AmazonEKSWorkerNodePolicy`.
    1. In addition to the managed policies add the following as inline policy. Replace _your-account-id_ with the actual value of the AWS account id you are using.
    
    
        ```{.bash}
        {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "VisualEditor0",
                "Effect": "Allow",
                "Action": [
                    "ec2:CreateVpc",
                    "ec2:DeleteVpc"
                ],
                "Resource": [
                    "arn:aws:ec2:*:your-account-id:ipv6pool-ec2/*",
                    "arn:aws:ec2::your-account-id:ipam-pool/*",
                    "arn:aws:ec2:*:your-account-id:vpc/*"
                ]
            },
            {
                "Sid": "VisualEditor1",
                "Effect": "Allow",
                "Action": [
                    "ec2:ModifyVpcAttribute",
                    "ec2:DescribeVpcAttribute"
                ],
                "Resource": "arn:aws:ec2:*:<your-account-id>:vpc/*"
            },
            {
                "Sid": "VisualEditor2",
                "Effect": "Allow",
                "Action": "ec2:AssociateVpcCidrBlock",
                "Resource": [
                    "arn:aws:ec2:*:your-account-id:ipv6pool-ec2/*",
                    "arn:aws:ec2::your-account-id:ipam-pool/*",
                    "arn:aws:ec2:*:your-account-id:vpc/*"
                ]
            },
            {
                "Sid": "VisualEditor3",
                "Effect": "Allow",
                "Action": [
                    "ec2:DescribeSecurityGroupRules",
                    "ec2:DescribeNatGateways",
                    "ec2:DescribeAddressesAttribute"
                ],
                "Resource": "*"
            },
            {
                "Sid": "VisualEditor4",
                "Effect": "Allow",
                "Action": [
                    "ec2:CreateInternetGateway",
                    "ec2:RevokeSecurityGroupEgress",
                    "ec2:CreateRouteTable",
                    "ec2:CreateSubnet"
                ],
                "Resource": [
                    "arn:aws:ec2:*:your-account-id:security-group/*",
                    "arn:aws:ec2:*:your-account-id:internet-gateway/*",
                    "arn:aws:ec2:*:your-account-id:subnet/*",
                    "arn:aws:ec2:*:your-account-id:route-table/*",
                    "arn:aws:ec2::your-account-id:ipam-pool/*",
                    "arn:aws:ec2:*:your-account-id:vpc/*"
                ]
            },
            {
                "Sid": "VisualEditor5",
                "Effect": "Allow",
                "Action": [
                    "ec2:AttachInternetGateway",
                    "ec2:AssociateRouteTable"
                ],
                "Resource": [
                    "arn:aws:ec2:*:your-account-id:vpn-gateway/*",
                    "arn:aws:ec2:*:your-account-id:internet-gateway/*",
                    "arn:aws:ec2:*:your-account-id:subnet/*",
                    "arn:aws:ec2:*:your-account-id:route-table/*",
                    "arn:aws:ec2:*:your-account-id:vpc/*"
                ]
            },
            {
                "Sid": "VisualEditor6",
                "Effect": "Allow",
                "Action": "ec2:AllocateAddress",
                "Resource": [
                    "arn:aws:ec2:*:your-account-id:ipv4pool-ec2/*",
                    "arn:aws:ec2:*:your-account-id:elastic-ip/*"
                ]
            },
            {
                "Sid": "VisualEditor7",
                "Effect": "Allow",
                "Action": "ec2:ReleaseAddress",
                "Resource": "arn:aws:ec2:*:your-account-id:elastic-ip/*"
            },
            {
                "Sid": "VisualEditor8",
                "Effect": "Allow",
                "Action": "ec2:CreateNatGateway",
                "Resource": [
                    "arn:aws:ec2:*:your-account-id:subnet/*",
                    "arn:aws:ec2:*:your-account-id:natgateway/*",
                    "arn:aws:ec2:*:your-account-id:elastic-ip/*"
                ]
            }
        ]
        }
        ```
1. Add the Role ARN and name here in the `variables.tf file by updating [these lines](https://github.com/awslabs/data-on-eks/blob/d532720d0746959daa6d3a3f5925fc8be114ccc4/ai-ml/trainium-inferentia/variables.tf#L126). Move the structure inside the `defaut` list and replace the role ARN and name values with the values for the role you are using.

1. Navigate into the `ai-ml/trainium-inferentia/` directory and run install.sh script.

    ``` {.bash}
    cd data-on-eks/ai-ml/trainium-inferentia/
    ./install.sh
    ```

    Note: This step takes about 12-15 minutes to deploy the EKS infrastructure and cluster in the AWS account. To view more details on cluster creation, view an example here: [Deploy Llama3 on EKS](https://awslabs.github.io/data-on-eks/docs/gen-ai/inference/llama3-inf2) in the _prerequisites_ section.
