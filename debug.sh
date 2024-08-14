# script for a debug/developer workflow
# 1. Deletes the existing pfmbench package from the conda env
# 2. Builds and installs a new one
# 3. Runs fmbench as usual

CONDA_ENV_NAME=fmbench_python311
CONDA_ENV_PATH=~/anaconda3/envs/$CONDA_ENV_NAME/lib/python3.11/site-packages
CONDA_ENV_PATH2=/home/sagemaker-user/.conda/envs/$CONDA_ENV_NAME/lib/python3.11/site-packages
CONFIG_FILE_PATH=src/fmbench/configs/bedrock/config-bedrock-llama3-1-no-streaming.yml
#src/fmbench/configs/bedrock/config-bedrock.yml
#src/fmbench/configs/llama3/8b/config-llama3-8b-g5-streaming.yml
#config-bedrock-llama3-streaming.yml #config-llama3-8b-g5-stream.yml
LOGFILE=fmbench.log

# delete existing install
rm -rf $CONDA_ENV_PATH/fmbench*
rm -rf $CONDA_ENV_PATH2/fmbench*


# build a new version
poetry build
pip install -U dist/*.whl

# run the newly installed version
echo "going to run fmbench now"
fmbench --config-file $CONFIG_FILE_PATH  > $LOGFILE 2>&1
echo "all done"