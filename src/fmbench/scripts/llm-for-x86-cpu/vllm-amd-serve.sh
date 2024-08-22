# Install Docker and Git using the YUM package manager
sudo yum install docker git -y

# Start the Docker service
sudo systemctl start docker

# Download the Miniconda installer for Linux
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \  # Run the Miniconda installer in batch mode (no manual intervention)
    && rm -f Miniconda3-latest-Linux-x86_64.sh \    # Remove the installer script after installation
    && eval "$(/home/$USER/miniconda3/bin/conda shell.bash hook)"\ # Initialize conda for bash shell
    && conda init  # Initialize conda, adding it to the shell

# Create a new conda environment named 'fmbench_python311' with Python 3.11 and ipykernel
conda create --name fmbench_python311 -y python=3.11 ipykernel

# Activate the newly created conda environment
source activate fmbench_python311

# Upgrade pip and install the fmbench package
pip install -U fmbench

# Clone the vLLM project repository from GitHub
git clone https://github.com/vllm-project/vllm.git

# Change the directory to the cloned vLLM project
cd vllm

# Build a Docker image using the provided Dockerfile for CPU, with a shared memory size of 4GB
sudo docker build -f Dockerfile.cpu -t vllm-cpu-env --shm-size=4g .

# Set the model ID for Meta Llama 3.1 and run the Docker container
# Replace 'your-hf-token-here' with your Hugging Face token to access the model
MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct
sudo docker run --rm --env "HF_TOKEN=your-hf-token-here" \  # Run Docker container with the specified Hugging Face token
  --ipc=host \  # Use the host's IPC namespace
  -p 8000:8000 \  # Map port 8000 of the container to port 8000 of the host
  -e VLLM_CPU_KVCACHE_SPACE=40 \  # Set the environment variable for CPU KV cache space
  vllm-cpu-env \  # Specify the Docker image to run
  --model $MODEL_ID  # Pass the model ID as an argument to the container

# Send a POST request to the running model server to generate a completion for a given prompt
curl --location 'http://localhost:8000/v1/completions' \  # Make a request to the local server
--header 'Content-Type: application/json' \  # Set the content type to JSON
--data '{ 
  "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",  # Specify the model to use
  "prompt": "tell me a story of the little red riding hood",  # Provide the prompt to generate text from
  "max_tokens": 100,  # Set the maximum number of tokens to generate
  "temperature": 0.1,  # Set the temperature for sampling (lower values make the output more deterministic)
  "top_p": 0.92,  # Set top-p sampling, the cumulative probability threshold for token selection
  "top_k": 120  # Set top-k sampling, the number of highest probability tokens to consider for each step
}'
