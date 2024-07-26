# script for a debug/developer workflow
# 1. Deletes the existing pfmbench package from the conda env
# 2. Builds and installs a new one
# 3. Runs fmbench as usual

CONDA_ENV_NAME=fmbench_python311
CONDA_ENV_PATH=~/anaconda3/envs/$CONDA_ENV_NAME/lib/python3.11/site-packages
CONFIG_FILE_PATH=src/fmbench/configs/llama2/7b/config-llama2-7b-g5-quick.yml
#src/fmbench/configs/llama3/8b/config-llama3-8b-g5-streaming.yml
#config-bedrock-llama3-streaming.yml #config-llama3-8b-g5-stream.yml
LOGFILE=fmbench.log

#pip uninstall fmbench -y
# delete existing install
rm -rf $CONDA_ENV_PATH/fmbench*

# build a new version
poetry build
pip install -U dist/*.whl

# run the newly installed version
echo "going to run fmbench now"
fmbench --config-file $CONFIG_FILE_PATH --local-mode yes --write-bucket abstaticwebsitetest1 > $LOGFILE 2>&1
echo "all done"
