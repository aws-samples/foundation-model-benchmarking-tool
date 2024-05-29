# script for a debug/developer workflow
# 1. Deletes the existing pfmbench package from the conda env
# 2. Builds and installs a new one
# 3. Runs fmbench as usual

CONDA_ENV_PATH=~/anaconda3/envs/fmbench_python311/lib/python3.11/site-packages
CONFIG_FILE_PATH=src/fmbench/configs/llama2/7b/config-llama2-7b-g5-no-s3-quick.yml
LOGFILE=fmbench.log

# delete existing install
rm -rf $CONDA_ENV_PATH/fmbench*

# build a new version
poetry build
pip install -U dist/*.whl

# run the newly installed version
echo "going to run fmbench now"
fmbench --config-file $CONFIG_FILE_PATH  > $LOGFILE 2>&1
echo "all done"
