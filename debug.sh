# script for a debug/developer workflow
# 1. Deletes the existing pfmbench package from the conda env
# 2. Builds and installs a new one
# 3. Runs fmbench as usual

CONDA_ENV_PATH=/Users/bainskb/opt/anaconda3/envs/fmbench_python311
CONFIG_FILE_PATH=src/fmbench/configs/mistral/config-mistral-7b-inf2.yml
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
