# Benchmark models on EC2

You can use `FMBench` to benchmark models on hosted on EC2. This can be done in one of two ways:

- Deploy the model on your EC2 instance independantly of `FMBench` and then benchmark it through the [Bring your own endpoint](#bring-your-own-endpoint-aka-support-for-external-endpoints) mode.
- Deploy the model on your EC2 instance through `FMBench` and then benchmark it.
 
The steps for deploying the model on your EC2 instance are described below. 

👉 In this configuration both the model being benchmarked and `FMBench` are deployed on the same EC2 instance.

Create a new EC2 instance suitable for hosting an LMI as per the steps described [here](misc/ec2_instance_creation_steps.md). _Note that you will need to select the correct AMI based on your instance type, this is called out in the instructions_.

## Benchmarking on an instance type with NVIDIA GPUs or AWS Chips

1. Connect to your instance using any of the options in EC2 (SSH/EC2 Connect), run the following in the EC2 terminal. This command installs Anaconda on the instance which is then used to create a new `conda` environment for `FMBench`. See instructions for downloading anaconda [here](https://www.anaconda.com/download)

    ```{.bash}
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b  # Run the Miniconda installer in batch mode (no manual intervention)
    rm -f Miniconda3-latest-Linux-x86_64.sh    # Remove the installer script after installation
    eval "$(/home/$USER/miniconda3/bin/conda shell.bash hook)" # Initialize conda for bash shell
    conda init  # Initialize conda, adding it to the shell  
    ```

1. Install `docker-compose`.

    ```{.bash}
    sudo apt-get update
    sudo apt-get install --reinstall docker.io -y
    sudo apt-get install -y docker-compose
    docker compose version 
    ```

1. Setup the `fmbench_python311` conda environment.

    ```{.bash}
    conda create --name fmbench_python311 -y python=3.11 ipykernel
    source activate fmbench_python311;
    pip install -U fmbench
    ```

1. Create local directory structure needed for `FMBench` and copy all publicly available dependencies from the AWS S3 bucket for `FMBench`. This is done by running the `copy_s3_content.sh` script available as part of the `FMBench` repo. Replace `/tmp` in the command below with a different path if you want to store the config files and the `FMBench` generated data in a different directory.

    ```{.bash}
    curl -s https://raw.githubusercontent.com/aws-samples/foundation-model-benchmarking-tool/main/copy_s3_content.sh | sh -s -- /tmp
    ```

1. To download the model files from HuggingFace, create a `hf_token.txt` file in the `/tmp/fmbench-read/scripts/` directory containing the Hugging Face token you would like to use. In the command below replace the `hf_yourtokenstring` with your Hugging Face token.

    ```{.bash}
    echo hf_yourtokenstring > /tmp/fmbench-read/scripts/hf_token.txt
    ```

1. Run `FMBench` with a packaged or a custom config file. **_This step will also deploy the model on the EC2 instance_**. The `--write-bucket` parameter value is just a placeholder and an actual S3 bucket is not required. **_Skip to the next step if benchmarking for AWS Chips_**. You could set the `--tmp-dir` flag to an EFA path instead of `/tmp` if using a shared path for storing config files and reports.

    ```{.bash}
    fmbench --config-file /tmp/fmbench-read/configs/llama3/8b/config-ec2-llama3-8b.yml --local-mode yes --write-bucket placeholder --tmp-dir /tmp > fmbench.log 2>&1
    ```

1. For example, to run `FMBench` on a `llama3-8b-Instruct` model on an `inf2.48xlarge` instance, run the command 
command below. The config file for this example can be viewed [here](src/fmbench/configs/llama3/8b/config-ec2-llama3-8b-inf2-48xl.yml).

    ```{.bash}
    fmbench --config-file /tmp/fmbench-read/configs/llama3/8b/config-ec2-llama3-8b-inf2-48xl.yml --local-mode yes --write-bucket placeholder --tmp-dir /tmp > fmbench.log 2>&1
    ```

1. Open a new Terminal and do a `tail` on `fmbench.log` to see a live log of the run.

    ```{.bash}
    tail -f fmbench.log
    ```

1. All metrics are stored in the `/tmp/fmbench-write` directory created automatically by the `fmbench` package. Once the run completes all files are copied locally in a `results-*` folder as usual.

## Benchmarking using a Triton Inference Server with custom Python Backend

**_As of 2024-09-10 this has been tested on a `trn1.32xlarge` instance_**

1. Connect to your instance using any of the options in EC2 (SSH/EC2 Connect), run the following in the EC2 terminal. This command installs Anaconda on the instance which is then used to create a new `conda` environment for `FMBench`. See instructions for downloading anaconda [here](https://www.anaconda.com/download)

    ```{.bash}
    # Install Docker and Git using the YUM package manager
    sudo yum install docker git -y

    # Start the Docker service
    sudo systemctl start docker

    # Download the Miniconda installer for Linux
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b  # Run the Miniconda installer in batch mode (no manual intervention)
    rm -f Miniconda3-latest-Linux-x86_64.sh    # Remove the installer script after installation
    eval "$(/home/$USER/miniconda3/bin/conda shell.bash hook)" # Initialize conda for bash shell
    conda init  # Initialize conda, adding it to the shell
    ```

1. Setup the `fmbench_python311` conda environment.

    ```{.bash}
    # Create a new conda environment named 'fmbench_python311' with Python 3.11 and ipykernel
    conda create --name fmbench_python311 -y python=3.11 ipykernel

    # Activate the newly created conda environment
    source activate fmbench_python311

    # Upgrade pip and install the fmbench package
    pip install -U fmbench
    ```

1. First we need to build the required docker image for `triton`, and push it locally:

    ```{.bash}
        git clone \
        https://github.com/aws-samples/amazon-eks-machine-learning-with-terraform-and-kubeflow.git
        cd amazon-eks-machine-learning-with-terraform-and-kubeflow/containers/tritonserver-neuronx/build_tools/
    ```

1. Create a script by copy pasting the commands and script below. This script will build and push the triton image locally

    ```{.bash}
        nano build_and_push_triton_fmbench.sh
    ```

    1. 👉 Copy paste the script below and execute the script:

        ```{.bash}
            #!/usr/bin/env bash

            # This script builds a Docker image and saves it locally in the home directory.

            # Set the image name and tag
            export IMAGE_NAME=tritonserver-neuronx
            export IMAGE_TAG=24.06-2.x

            # Get the directory of the current script
            DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

            # Build the Docker image locally with the image name
            docker build -t ${IMAGE_NAME}:${IMAGE_TAG} ${DIR}/..

            if [ $? -ne 0 ]; then
                echo "Error: Docker image build failed"
                exit 1
            fi

            # Save the Docker image locally in the home directory
            HOME_DIR=$(eval echo ~$USER)
            docker save ${IMAGE_NAME}:${IMAGE_TAG} | gzip > ${HOME_DIR}/${IMAGE_NAME}_${IMAGE_TAG}.tar.gz

            if [ $? -ne 0 ]; then
                echo "Error: Failed to save Docker image"
                exit 1
            else
                echo "Triton docker image saved as ${IMAGE_NAME}_${IMAGE_TAG}.tar.gz in the home directory (${HOME_DIR})."
            fi
        ```
    
    1. Make the script executable and run it:

        ```{.bash}
            chmod +x build_and_push_triton_fmbench.sh
            ./build_and_push_triton_fmbench.sh
        ```

        - Now wait until the docker image is saved locally and then follow the instructions below to start a benchmarking test.

1. Create local directory structure needed for `FMBench` and copy all publicly available dependencies from the AWS S3 bucket for `FMBench`. This is done by running the `copy_s3_content.sh` script available as part of the `FMBench` repo. Replace `/tmp` in the command below with a different path if you want to store the config files and the `FMBench` generated data in a different directory.

    ```{.bash}
    curl -s https://raw.githubusercontent.com/aws-samples/foundation-model-benchmarking-tool/main/copy_s3_content.sh | sh -s -- /tmp
    ```

1. To download the model files from HuggingFace, create a `hf_token.txt` file in the `/tmp/fmbench-read/scripts/` directory containing the Hugging Face token you would like to use. In the command below replace the `hf_yourtokenstring` with your Hugging Face token.

    ```{.bash}
    echo hf_yourtokenstring > /tmp/fmbench-read/scripts/hf_token.txt
    ```

1. Run `FMBench` with a packaged or a custom config file. **_This step will also deploy the model on the EC2 instance_**. The `--write-bucket` parameter value is just a placeholder and an actual S3 bucket is not required. You could set the `--tmp-dir` flag to an EFA path instead of `/tmp` if using a shared path for storing config files and reports.

    ```{.bash}
    fmbench --config-file <tmp/path/to/be/added> --local-mode yes --write-bucket placeholder --tmp-dir /tmp > fmbench.log 2>&1
    ```

1. Open a new Terminal and and do a `tail` on `fmbench.log` to see a live log of the run.

    ```{.bash}
    tail -f fmbench.log
    ```

1. All metrics are stored in the `/tmp/fmbench-write` directory created automatically by the `fmbench` package. Once the run completes all files are copied locally in a `results-*` folder as usual.



## Benchmarking on an CPU instance type with AMD processors

**_As of 2024-08-27 this has been tested on a `m7a.16xlarge` instance_**

1. Connect to your instance using any of the options in EC2 (SSH/EC2 Connect), run the following in the EC2 terminal. This command installs Anaconda on the instance which is then used to create a new `conda` environment for `FMBench`. See instructions for downloading anaconda [here](https://www.anaconda.com/download)

    ```{.bash}
    # Install Docker and Git using the YUM package manager
    sudo yum install docker git -y

    # Start the Docker service
    sudo systemctl start docker

    # Download the Miniconda installer for Linux
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b  # Run the Miniconda installer in batch mode (no manual intervention)
    rm -f Miniconda3-latest-Linux-x86_64.sh    # Remove the installer script after installation
    eval "$(/home/$USER/miniconda3/bin/conda shell.bash hook)" # Initialize conda for bash shell
    conda init  # Initialize conda, adding it to the shell
    ```

1. Setup the `fmbench_python311` conda environment.

    ```{.bash}
    # Create a new conda environment named 'fmbench_python311' with Python 3.11 and ipykernel
    conda create --name fmbench_python311 -y python=3.11 ipykernel

    # Activate the newly created conda environment
    source activate fmbench_python311

    # Upgrade pip and install the fmbench package
    pip install -U fmbench
    ```

1. Build the `vllm` container for serving the model. 

    1. 👉 The `vllm` container we are building locally is going to be references in the `FMBench` config file.

    1. The container being build is for CPU only (GPU support might be added in future).

        ```{.bash}
        # Clone the vLLM project repository from GitHub
        git clone https://github.com/vllm-project/vllm.git

        # Change the directory to the cloned vLLM project
        cd vllm

        # Build a Docker image using the provided Dockerfile for CPU, with a shared memory size of 4GB
        sudo docker build -f Dockerfile.cpu -t vllm-cpu-env --shm-size=4g .
        ```

1. Create local directory structure needed for `FMBench` and copy all publicly available dependencies from the AWS S3 bucket for `FMBench`. This is done by running the `copy_s3_content.sh` script available as part of the `FMBench` repo. Replace `/tmp` in the command below with a different path if you want to store the config files and the `FMBench` generated data in a different directory.

    ```{.bash}
    curl -s https://raw.githubusercontent.com/aws-samples/foundation-model-benchmarking-tool/main/copy_s3_content.sh | sh -s -- /tmp
    ```

1. To download the model files from HuggingFace, create a `hf_token.txt` file in the `/tmp/fmbench-read/scripts/` directory containing the Hugging Face token you would like to use. In the command below replace the `hf_yourtokenstring` with your Hugging Face token.

    ```{.bash}
    echo hf_yourtokenstring > /tmp/fmbench-read/scripts/hf_token.txt
    ```

1. Before running FMBench, add the current user to the docker group. Run the following commands to run Docker without needing to use `sudo` each time.

    ```{.bash}
    sudo usermod -a -G docker $USER
    newgrp docker
    ```

1. Install `docker-compose`.

    ```{.bash}
    DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
    mkdir -p $DOCKER_CONFIG/cli-plugins
    sudo curl -L https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m) -o $DOCKER_CONFIG/cli-plugins/docker-compose
    sudo chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose
    docker compose version
    ```

1. Run `FMBench` with a packaged or a custom config file. **_This step will also deploy the model on the EC2 instance_**. The `--write-bucket` parameter value is just a placeholder and an actual S3 bucket is not required. You could set the `--tmp-dir` flag to an EFA path instead of `/tmp` if using a shared path for storing config files and reports.

    ```{.bash}
    fmbench --config-file /tmp/fmbench-read/configs/llama3/8b/config-ec2-llama3-8b-m7a-16xlarge.yml --local-mode yes --write-bucket placeholder --tmp-dir /tmp > fmbench.log 2>&1
    ```

1. Open a new Terminal and and do a `tail` on `fmbench.log` to see a live log of the run.

    ```{.bash}
    tail -f fmbench.log
    ```

1. All metrics are stored in the `/tmp/fmbench-write` directory created automatically by the `fmbench` package. Once the run completes all files are copied locally in a `results-*` folder as usual.


## Benchmarking on an CPU instance type with Intel processors

**_As of 2024-08-27 this has been tested on `c5.18xlarge` and `m5.16xlarge` instances_**

1. Connect to your instance using any of the options in EC2 (SSH/EC2 Connect), run the following in the EC2 terminal. This command installs Anaconda on the instance which is then used to create a new `conda` environment for `FMBench`. See instructions for downloading anaconda [here](https://www.anaconda.com/download)

    ```{.bash}
    # Install Docker and Git using the YUM package manager
    sudo yum install docker git -y

    # Start the Docker service
    sudo systemctl start docker

    # Download the Miniconda installer for Linux
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b # Run the Miniconda installer in batch mode (no manual intervention)
    rm -f Miniconda3-latest-Linux-x86_64.sh    # Remove the installer script after installation
    eval "$(/home/$USER/miniconda3/bin/conda shell.bash hook)" # Initialize conda for bash shell
    conda init  # Initialize conda, adding it to the shell
    ```

1. Setup the `fmbench_python311` conda environment.

    ```{.bash}
    # Create a new conda environment named 'fmbench_python311' with Python 3.11 and ipykernel
    conda create --name fmbench_python311 -y python=3.11 ipykernel

    # Activate the newly created conda environment
    source activate fmbench_python311

    # Upgrade pip and install the fmbench package
    pip install -U fmbench
    ```

1. Build the `vllm` container for serving the model. 

    1. 👉 The `vllm` container we are building locally is going to be references in the `FMBench` config file.

    1. The container being build is for CPU only (GPU support might be added in future).

        ```{.bash}
        # Clone the vLLM project repository from GitHub
        git clone https://github.com/vllm-project/vllm.git

        # Change the directory to the cloned vLLM project
        cd vllm

        # Build a Docker image using the provided Dockerfile for CPU, with a shared memory size of 12GB
        sudo docker build -f Dockerfile.cpu -t vllm-cpu-env --shm-size=12g .
        ```

1. Create local directory structure needed for `FMBench` and copy all publicly available dependencies from the AWS S3 bucket for `FMBench`. This is done by running the `copy_s3_content.sh` script available as part of the `FMBench` repo. Replace `/tmp` in the command below with a different path if you want to store the config files and the `FMBench` generated data in a different directory.

    ```{.bash}
    curl -s https://raw.githubusercontent.com/aws-samples/foundation-model-benchmarking-tool/main/copy_s3_content.sh | sh -s -- /tmp
    ```

1. To download the model files from HuggingFace, create a `hf_token.txt` file in the `/tmp/fmbench-read/scripts/` directory containing the Hugging Face token you would like to use. In the command below replace the `hf_yourtokenstring` with your Hugging Face token.

    ```{.bash}
    echo hf_yourtokenstring > /tmp/fmbench-read/scripts/hf_token.txt
    ```

1. Before running FMBench, add the current user to the docker group. Run the following commands to run Docker without needing to use `sudo` each time.

    ```{.bash}
    sudo usermod -a -G docker $USER
    newgrp docker
    ```

1. Install `docker-compose`.

    ```{.bash}
    DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
    mkdir -p $DOCKER_CONFIG/cli-plugins
    sudo curl -L https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m) -o $DOCKER_CONFIG/cli-plugins/docker-compose
    sudo chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose
    docker compose version
    ```

1. Run `FMBench` with a packaged or a custom config file. **_This step will also deploy the model on the EC2 instance_**. The `--write-bucket` parameter value is just a placeholder and an actual S3 bucket is not required. You could set the `--tmp-dir` flag to an EFA path instead of `/tmp` if using a shared path for storing config files and reports.

    ```{.bash}
    fmbench --config-file /tmp/fmbench-read/configs/llama3/8b/config-ec2-llama3-8b-c5-18xlarge.yml --local-mode yes --write-bucket placeholder --tmp-dir /tmp > fmbench.log 2>&1
    ```

1. Open a new Terminal and and do a `tail` on `fmbench.log` to see a live log of the run.

    ```{.bash}
    tail -f fmbench.log
    ```

1. All metrics are stored in the `/tmp/fmbench-write` directory created automatically by the `fmbench` package. Once the run completes all files are copied locally in a `results-*` folder as usual.