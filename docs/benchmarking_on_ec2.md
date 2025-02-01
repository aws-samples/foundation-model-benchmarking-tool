# Benchmark models on EC2

You can use `FMBench` to benchmark models on hosted on EC2. This can be done in one of two ways:

- Deploy the model on your EC2 instance independently of `FMBench` and then benchmark it through the [Bring your own endpoint](#bring-your-own-endpoint-aka-support-for-external-endpoints) mode.
- Deploy the model on your EC2 instance through `FMBench` and then benchmark it.
 
The steps for deploying the model on your EC2 instance are described below. 

👉 In this configuration both the model being benchmarked and `FMBench` are deployed on the same EC2 instance.

Create a new EC2 instance suitable for hosting an LMI as per the steps described [here](misc/ec2_instance_creation_steps.md). _Note that you will need to select the correct AMI based on your instance type, this is called out in the instructions_.

The steps for benchmarking on different types of EC2 instances (GPU/CPU/Neuron) and different inference containers differ slightly. These are all described below.

## Benchmarking options on EC2
- [Benchmarking on an instance type with NVIDIA GPUs or AWS Chips](#benchmarking-on-an-instance-type-with-nvidia-gpus-or-aws-chips)
- [Benchmarking on an instance type with NVIDIA GPU and the Triton inference server](#benchmarking-on-an-instance-type-with-nvidia-gpu-and-the-triton-inference-server)
- [Benchmarking on an instance type with AWS Chips and the Triton inference server](#benchmarking-on-an-instance-type-with-aws-chips-and-the-triton-inference-server)
- [Benchmarking on an CPU instance type with AMD processors](#benchmarking-on-an-cpu-instance-type-with-amd-processors)
- [Benchmarking on an CPU instance type with Intel processors](#benchmarking-on-an-cpu-instance-type-with-intel-processors)
- [Benchmarking on an CPU instance type with ARM processors (Graviton 4)](#benchmarking-on-an-cpu-instance-type-with-arm-processors)

- [Benchmarking the Triton inference server](#benchmarking-the-triton-inference-server)
- [Benchmarking models on Ollama](#benchmarking-models-on-ollama)

## Benchmarking on an instance type with NVIDIA GPUs or AWS Chips

1. Connect to your instance using any of the options in EC2 (SSH/EC2 Connect), run the following in the EC2 terminal. This command installs `uv` on the instance which is then used to create a new virtual environment for `FMBench`.

    ```{.bash}
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    ```

1. Install `docker-compose`.

    ```{.bash}
    sudo apt-get update
    sudo apt-get install --reinstall docker.io -y
    sudo apt-get install -y docker-compose
    docker compose version 
    ```

1. Setup the `.fmbench_python311` Python environment.

    ```{.bash}
    uv venv .fmbench_python311 --python 3.11
    source .fmbench_python311/bin/activate
    # Add the Python environment activation and directory navigation to .bashrc
    echo 'source $HOME/.fmbench_python311/bin/activate' >> $HOME/.bashrc
    uv pip install -U fmbench
    ```

1. Create local directory structure needed for `FMBench` and copy all publicly available dependencies from the AWS S3 bucket for `FMBench`. This is done by running the `copy_s3_content.sh` script available as part of the `FMBench` repo. **Replace `/tmp` in the command below with a different path if you want to store the config files and the `FMBench` generated data in a different directory**.

    ```{.bash}
    # Replace "/tmp" with "/path/to/your/custom/tmp" if you want to use a custom tmp directory
    TMP_DIR="/tmp"
    curl -s https://raw.githubusercontent.com/aws-samples/foundation-model-benchmarking-tool/main/copy_s3_content.sh | sh -s -- "$TMP_DIR"
    ```

1. To download the model files from HuggingFace, create a `hf_token.txt` file in the `/tmp/fmbench-read/scripts/` directory containing the Hugging Face token you would like to use. In the command below replace the `hf_yourtokenstring` with your Hugging Face token. **Replace `/tmp` in the command below if you are using `/path/to/your/custom/tmp` to store the config files and the `FMBench` generated data**.

    ```{.bash}
    echo hf_yourtokenstring > $TMP_DIR/fmbench-read/scripts/hf_token.txt
    ```

1. Run `FMBench` with a packaged or a custom config file. **_This step will also deploy the model on the EC2 instance_**. The `--write-bucket` parameter value is just a placeholder and an actual S3 bucket is not required. **_Skip to the next step if benchmarking for AWS Chips_**. You could set the `--tmp-dir` flag to an EFA path instead of `/tmp` if using a shared path for storing config files and reports.

    ```{.bash}
    fmbench --config-file $TMP_DIR/fmbench-read/configs/llama3/8b/config-ec2-llama3-8b.yml --local-mode yes --write-bucket placeholder --tmp-dir $TMP_DIR > fmbench.log 2>&1
    ```

1. For example, to run `FMBench` on a `llama3-8b-Instruct` model on an `inf2.48xlarge` instance, run the command 
command below. The config file for this example can be viewed [here](src/fmbench/configs/llama3/8b/config-ec2-llama3-8b-inf2-48xl.yml).

    ```{.bash}
    fmbench --config-file $TMP_DIR/fmbench-read/configs/llama3/8b/config-ec2-llama3-8b-inf2-48xl.yml --local-mode yes --write-bucket placeholder --tmp-dir $TMP_DIR > fmbench.log 2>&1
    ```

1. Open a new Terminal and do a `tail` on `fmbench.log` to see a live log of the run.

    ```{.bash}
    tail -f fmbench.log
    ```

1. All metrics are stored in the `/tmp/fmbench-write` directory created automatically by the `fmbench` package. Once the run completes all files are copied locally in a `results-*` folder as usual.

## Benchmarking on an instance type with NVIDIA GPU and the Triton inference server

1. Follow steps in the [Benchmarking on an instance type with NVIDIA GPUs or AWS Chips](#benchmarking-on-an-instance-type-with-nvidia-gpus-or-aws-chips) section to install `FMBench` but do not run any benchmarking tests yet.

1. Once `FMBench` is installed then install the following additional dependencies for Triton.

    ```{.bash}
    cd ~
    git clone https://github.com/triton-inference-server/tensorrtllm_backend.git  --branch v0.12.0
    # Update the submodules
    cd tensorrtllm_backend
    # Install git-lfs if needed
    sudo apt --fix-broken install
    sudo apt-get update && sudo apt-get install git-lfs -y --no-install-recommends
    git lfs install
    git submodule update --init --recursive
    ```

1. Now you are ready to run benchmarking with Triton. For example for benchmarking `Llama3-8b` model on a `g5.12xlarge` use the following command:

    ```{.bash}
    fmbench --config-file $TMP_DIR/fmbench-read/configs/llama3/8b/config-llama3-8b-g5.12xl-tp-2-mc-max-triton-ec2.yml --local-mode yes --write-bucket placeholder --tmp-dir $TMP_DIR > fmbench.log 2>&1
    ```

## Benchmarking on an instance type with AWS Chips and the Triton inference server

**_As of 2024-09-26 this has been tested on a `trn1.32xlarge` instance_**

1. Connect to your instance using any of the options in EC2 (SSH/EC2 Connect), run the following in the EC2 terminal. This command installs `uv` on the instance which is then used to create a new Python virtual environment for `FMBench`.(Note: **_Your EC2 instance needs to have at least 200GB of disk space for this test_**)

    ```{.bash}
    # Install Docker and Git using the YUM package manager
    sudo yum install docker git -y

    # Start the Docker service
    sudo systemctl start docker

    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    ```

1. Setup the `.fmbench_python311` Python virtual environment.

    ```{.bash}
    uv venv .fmbench_python311 --python 3.11
    source .fmbench_python311/bin/activate
    # Add the Python environment activation and directory navigation to .bashrc
    echo 'source $HOME/.fmbench_python311/bin/activate' >> $HOME/.bashrc
    uv pip install -U fmbench
    ```

1. First we need to build the required docker image for `triton`, and push it locally. To do this, curl the `Triton Dockerfile` and the script to build and push the triton image locally:

    ```{.bash}
    # curl the docker file for triton
    curl -o ./Dockerfile_triton https://raw.githubusercontent.com/aws-samples/foundation-model-benchmarking-tool/main/src/fmbench/scripts/triton/Dockerfile_triton

    # curl the script that builds and pushes the triton image locally
    curl -o build_and_push_triton.sh https://raw.githubusercontent.com/aws-samples/foundation-model-benchmarking-tool/main/src/fmbench/scripts/triton/build_and_push_triton.sh

    # Make the triton build and push script executable, and run it
    chmod +x build_and_push_triton.sh
    ./build_and_push_triton.sh
    ```
   - Now wait until the docker image is saved locally and then follow the instructions below to start a benchmarking test.

1. Create local directory structure needed for `FMBench` and copy all publicly available dependencies from the AWS S3 bucket for `FMBench`. This is done by running the `copy_s3_content.sh` script available as part of the `FMBench` repo. **Replace `/tmp` in the command below with a different path if you want to store the config files and the `FMBench` generated data in a different directory**.

    ```{.bash}
    # Replace "/tmp" with "/path/to/your/custom/tmp" if you want to use a custom tmp directory
    TMP_DIR="/tmp"
    curl -s https://raw.githubusercontent.com/aws-samples/foundation-model-benchmarking-tool/main/copy_s3_content.sh | sh -s -- "$TMP_DIR"
    ```

1. To download the model files from HuggingFace, create a `hf_token.txt` file in the `/tmp/fmbench-read/scripts/` directory containing the Hugging Face token you would like to use. In the command below replace the `hf_yourtokenstring` with your Hugging Face token. **Replace `/tmp` in the command below if you are using `/path/to/your/custom/tmp` to store the config files and the `FMBench` generated data**.

    ```{.bash}
    echo hf_yourtokenstring > $TMP_DIR/fmbench-read/scripts/hf_token.txt
    ```

1. Run `FMBench` with a packaged or a custom config file. **_This step will also deploy the model on the EC2 instance_**. The `--write-bucket` parameter value is just a placeholder and an actual S3 bucket is not required. You could set the `--tmp-dir` flag to an EFA path instead of `/tmp` if using a shared path for storing config files and reports. 

    ```{.bash}
    fmbench --config-file $TMP_DIR/fmbench-read/configs/llama3/8b/config-llama3-8b-trn1-32xlarge-triton-djl.yml --local-mode yes --write-bucket placeholder --tmp-dir $TMP_DIR > fmbench.log 2>&1
    ```

1. Open a new Terminal and and do a `tail` on `fmbench.log` to see a live log of the run.

    ```{.bash}
    tail -f fmbench.log
    ```

1. All metrics are stored in the `/tmp/fmbench-write` directory created automatically by the `fmbench` package. Once the run completes all files are copied locally in a `results-*` folder as usual.

 - **Note**: To deploy a model on AWS Chips using Triton with `djl` or `vllm` backend, the configuration file requires the `backend` and `container_params` parameters within the `inference_spec` dictionary. The backend options are `vllm`/`djl` and the `container_params` contains container specific parameters to deploy the model, for example `tensor parallel degree`, `n positions`, etc. Tensor parallel degree is a necessary field to be added. If no other parameters are provided, the container will choose the default parameters during deployment.

    ``` {.python}
      # Backend options: [djl, vllm]
      backend: djl

      # Container parameters that are used during model deployment
      container_params:
        # tp degree is a mandatory parameter
        tp_degree: 8
        amp: "f16"
        attention_layout: 'BSH'
        collectives_layout: 'BSH'
        context_length_estimate: 3072, 3584, 4096
        max_rolling_batch_size: 8
        model_loader: "tnx"
        model_loading_timeout: 2400
        n_positions: 4096
        output_formatter: "json"
        rolling_batch: "auto"
        rolling_batch_strategy: "continuous_batching"
        trust_remote_code: true
        # modify the serving properties to match your model and requirements
        serving.properties:
    ```

## Benchmarking on an CPU instance type with AMD processors

**_As of 2024-08-27 this has been tested on a `m7a.16xlarge` instance_**

1. Connect to your instance using any of the options in EC2 (SSH/EC2 Connect), run the following in the EC2 terminal. This command installs `uv` on the instance which is then used to create a new Python virtual environment for `FMBench`.

    ```{.bash}
    # Install Docker and Git using the YUM package manager
    sudo yum install docker git -y

    # Start the Docker service
    sudo systemctl start docker

    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    ```

1. Setup the `.fmbench_python311` Python virtual environment.

    ```{.bash}
    uv venv .fmbench_python311 --python 3.11
    source .fmbench_python311/bin/activate
    # Add the Python environment activation and directory navigation to .bashrc
    echo 'source $HOME/.fmbench_python311/bin/activate' >> $HOME/.bashrc
    uv pip install -U fmbench
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

1. Create local directory structure needed for `FMBench` and copy all publicly available dependencies from the AWS S3 bucket for `FMBench`. This is done by running the `copy_s3_content.sh` script available as part of the `FMBench` repo. **Replace `/tmp` in the command below with a different path if you want to store the config files and the `FMBench` generated data in a different directory**.

    ```{.bash}
    # Replace "/tmp" with "/path/to/your/custom/tmp" if you want to use a custom tmp directory
    TMP_DIR="/tmp"
    curl -s https://raw.githubusercontent.com/aws-samples/foundation-model-benchmarking-tool/main/copy_s3_content.sh | sh -s -- "$TMP_DIR"
    ```

1. To download the model files from HuggingFace, create a `hf_token.txt` file in the `/tmp/fmbench-read/scripts/` directory containing the Hugging Face token you would like to use. In the command below replace the `hf_yourtokenstring` with your Hugging Face token. **Replace `/tmp` in the command below if you are using `/path/to/your/custom/tmp` to store the config files and the `FMBench` generated data**.

    ```{.bash}
    echo hf_yourtokenstring > $TMP_DIR/fmbench-read/scripts/hf_token.txt
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
    fmbench --config-file $TMP_DIR/fmbench-read/configs/llama3/8b/config-ec2-llama3-8b-m7a-16xlarge.yml --local-mode yes --write-bucket placeholder --tmp-dir $TMP_DIR > fmbench.log 2>&1
    ```

1. Open a new Terminal and and do a `tail` on `fmbench.log` to see a live log of the run.

    ```{.bash}
    tail -f fmbench.log
    ```

1. All metrics are stored in the `/tmp/fmbench-write` directory created automatically by the `fmbench` package. Once the run completes all files are copied locally in a `results-*` folder as usual.


## Benchmarking on an CPU instance type with Intel processors

**_As of 2024-08-27 this has been tested on `c5.18xlarge` and `m5.16xlarge` instances_**

1. Connect to your instance using any of the options in EC2 (SSH/EC2 Connect), run the following in the EC2 terminal. This command installs `uv` on the instance which is then used to create a new Python virtual environment for `FMBench`.

    ```{.bash}
    # Install Docker and Git using the YUM package manager
    sudo yum install docker git -y

    # Start the Docker service
    sudo systemctl start docker

    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    ```

1. Setup the `.fmbench_python311` Python virtual environment.

    ```{.bash}
    uv venv .fmbench_python311 --python 3.11
    source .fmbench_python311/bin/activate
    # Add the Python environment activation and directory navigation to .bashrc
    echo 'source $HOME/.fmbench_python311/bin/activate' >> $HOME/.bashrc
    uv pip install -U fmbench
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

1. Create local directory structure needed for `FMBench` and copy all publicly available dependencies from the AWS S3 bucket for `FMBench`. This is done by running the `copy_s3_content.sh` script available as part of the `FMBench` repo. **Replace `/tmp` in the command below with a different path if you want to store the config files and the `FMBench` generated data in a different directory**.

    ```{.bash}
    # Replace "/tmp" with "/path/to/your/custom/tmp" if you want to use a custom tmp directory
    TMP_DIR="/tmp"
    curl -s https://raw.githubusercontent.com/aws-samples/foundation-model-benchmarking-tool/main/copy_s3_content.sh | sh -s -- "$TMP_DIR"
    ```

1. To download the model files from HuggingFace, create a `hf_token.txt` file in the `/tmp/fmbench-read/scripts/` directory containing the Hugging Face token you would like to use. In the command below replace the `hf_yourtokenstring` with your Hugging Face token. **Replace `/tmp` in the command below if you are using `/path/to/your/custom/tmp` to store the config files and the `FMBench` generated data**.

    ```{.bash}
    echo hf_yourtokenstring > $TMP_DIR/fmbench-read/scripts/hf_token.txt
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
    fmbench --config-file $TMP_DIR/fmbench-read/configs/llama3/8b/config-ec2-llama3-8b-c5-18xlarge.yml --local-mode yes --write-bucket placeholder --tmp-dir $TMP_DIR > fmbench.log 2>&1
    ```

1. Open a new Terminal and and do a `tail` on `fmbench.log` to see a live log of the run.

    ```{.bash}
    tail -f fmbench.log
    ```

1. All metrics are stored in the `/tmp/fmbench-write` directory created automatically by the `fmbench` package. Once the run completes all files are copied locally in a `results-*` folder as usual.


## Benchmarking models on Ollama

**_As of 10/24/2024, this has been tested on `g6e.2xlarge` with `llama 3.1 8b`_**

1. Install Ollama.

    ```{bash}

    curl -fsSL https://ollama.com/install.sh | sh

    ```

1. Pull the model required.

    ```{bash}

    ollama pull llama3.1:8b

    ```

1. Serve the model. This might produce the following error message: `Error: accepts 0 arg(s), received 1` but you can safely ignore this error.

    ```{bash}

    ollama serve llama3.1:8b

    ```
    
1. Create local directory structure needed for `FMBench` and copy all publicly available dependencies from the AWS S3 bucket for `FMBench`. This is done by running the `copy_s3_content.sh` script available as part of the `FMBench` repo. **Replace `/tmp` in the command below with a different path if you want to store the config files and the `FMBench` generated data in a different directory**.

    ```{.bash}
    # Replace "/tmp" with "/path/to/your/custom/tmp" if you want to use a custom tmp directory
    TMP_DIR="/tmp"
    curl -s https://raw.githubusercontent.com/aws-samples/foundation-model-benchmarking-tool/main/copy_s3_content.sh | sh -s -- "$TMP_DIR"
    ```


1. Run `FMBench` with a packaged or a custom config file. The `--write-bucket` parameter value is just a placeholder and an actual S3 bucket is not required. You could set the `--tmp-dir` flag to an EFA path instead of `/tmp` if using a shared path for storing config files and reports.

    ```{.bash}
    fmbench --config-file $TMP_DIR/fmbench-read/configs/llama3.1/8b/config-ec2-llama3-1-8b-g6e-2xlarge-byoe-ollama.yml --local-mode yes --write-bucket placeholder --tmp-dir $TMP_DIR > fmbench.log 2>&1
    ```


## Benchmarking on an CPU instance type with ARM processors

**_As of 12/24/2024, this has been tested on `c8g.24xlarge` with `llama 3 8b Instruct` on Ubuntu Server 24.04 LTS (HVM), SSD Volume Type_**


1. Connect to your instance using any of the options in EC2 (SSH/EC2 Connect), run the following in the EC2 terminal. This command installs `Docker` and `uv` on the instance which is then used to create a new Python virtual environment for `FMBench`.

    ```{.bash}

    sudo apt-get update -y
    sudo apt-get install -y docker.io git
    sudo systemctl start docker
    sudo systemctl enable docker

    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    ```

1. Setup the `.fmbench_python311` Python virtual environment.

    ```{.bash}
    uv venv .fmbench_python311 --python 3.11
    source .fmbench_python311/bin/activate
    # Add the Python environment activation and directory navigation to .bashrc
    echo 'source $HOME/.fmbench_python311/bin/activate' >> $HOME/.bashrc
    uv pip install -U fmbench
    ```

1. Build the `vllm` container for serving the model. 

    1. 👉 The `vllm` container we are building locally is going to be referenced in the `FMBench` config file.

    1. The container being built is for ARM CPUs only.

        ```{.bash}
        # Clone the vLLM project repository from GitHub
        git clone https://github.com/vllm-project/vllm.git

        # Change the directory to the cloned vLLM project
        cd vllm

        # Build a Docker image using the provided Dockerfile for CPU, with a shared memory size of 12GB
        sudo docker build -f Dockerfile.arm -t vllm-cpu-env --shm-size=12g .
        ```

1. Create local directory structure needed for `FMBench` and copy all publicly available dependencies from the AWS S3 bucket for `FMBench`. This is done by running the `copy_s3_content.sh` script available as part of the `FMBench` repo. **Replace `/tmp` in the command below with a different path if you want to store the config files and the `FMBench` generated data in a different directory**.

    ```{.bash}
    # Replace "/tmp" with "/path/to/your/custom/tmp" if you want to use a custom tmp directory
    TMP_DIR="/tmp"
    curl -s https://raw.githubusercontent.com/aws-samples/foundation-model-benchmarking-tool/main/copy_s3_content.sh | sh -s -- "$TMP_DIR"
    ```

1. To download the model files from HuggingFace, create a `hf_token.txt` file in the `/tmp/fmbench-read/scripts/` directory containing the Hugging Face token you would like to use. In the command below replace the `hf_yourtokenstring` with your Hugging Face token. **Replace `/tmp` in the command below if you are using `/path/to/your/custom/tmp` to store the config files and the `FMBench` generated data**.

    ```{.bash}
    echo hf_yourtokenstring > $TMP_DIR/fmbench-read/scripts/hf_token.txt
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
    fmbench --config-file $TMP_DIR/fmbench-read/configs/llama3/8b/config-ec2-llama3-8b-c8g-24xlarge.yml --local-mode yes --write-bucket placeholder --tmp-dir $TMP_DIR > fmbench.log 2>&1
    ```

1. Open a new Terminal and and do a `tail` on `fmbench.log` to see a live log of the run.

    ```{.bash}
    tail -f fmbench.log
    ```

1. All metrics are stored in the `/tmp/fmbench-write` directory created automatically by the `fmbench` package. Once the run completes all files are copied locally in a `results-*` folder as usual.
