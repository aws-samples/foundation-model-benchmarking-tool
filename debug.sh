# script for a debug/developer workflow
# 1. Deletes the existing pfmbench package from the conda env
# 2. Builds and installs a new one
# 3. Runs fmbench as usual

CONDA_ENV_PATH=$CONDA_PREFIX/lib/python3.11/site-packages
CONFIG_FILE_PATH=src/fmbench/configs/llama3/8b/config-ec2-llama3-8b.yml
#src/fmbench/configs/llama3.1/8b/config-llama3.1-8b-g5.yml
#src/fmbench/configs/llama3/8b/config-ec2-llama3-8b-m7a-16xlarge.yml
#src/fmbench/configs/mistral/config-mistral-v3-inf2-48xl-deploy-ec2-tp24.yml
#bedrock/config-bedrock-llama3-1-no-streaming.yml
#src/fmbench/configs/bedrock/config-bedrock.yml
#src/fmbench/configs/llama3/8b/config-llama3-8b-g5-streaming.yml
#config-bedrock-llama3-streaming.yml #config-llama3-8b-g5-stream.yml
LOGFILE=fmbench.log

# delete existing install
rm -rf $CONDA_ENV_PATH/fmbench*

# build a new version
poetry build
pip install -U dist/*.whl

# run the newly installed version
echo "going to run fmbench now"
# fmbench --config-file $CONFIG_FILE_PATH  --local-mode yes --write-bucket placeholder --tmp-dir /tmp> $LOGFILE 2>&1

# Use FMBench to benchmark models on hosted on EC2 using the command below. If you want to write the metrics and results to an
# s3 bucket, replace `placeholder` with the name of that s3 bucket in your AWS account. Optionally, you can send the results to
# a custom tmp directory by setting the '--tmp-dir' argument followed by the path to that custom tmp directory. If '--tmp-dir' is not
# provided, the default 'tmp' directory will be used.
fmbench --config-file $CONFIG_FILE_PATH --local-mode yes --write-bucket placeholder --tmp-dir /tmp > $LOGFILE 2>&1
echo "all done"
