#!/bin/bash
[ ! -d /triton ] && echo "/triton dir must exist" && exit 1
[ $# -ne 4 ] && echo "usage: $0 hf-model-id model-name http-port opm-num-threads" && exit 1

# The HF model id, model name and http port
# are parsed from the triton multi model preparation
# function here
HF_MODEL_ID=$1
MODEL_NAME=$2
HTTP_PORT=$3
OPM_NUM_THREADS=$4

LOG_ROOT=/triton/logs
MODEL_REPO=/triton/model_repository
CACHE_DIR=/cache

GIT_CLONE_DIR=/tmp/vllm
git clone https://github.com/vllm-project/vllm.git $GIT_CLONE_DIR
cd $GIT_CLONE_DIR
git checkout main
git fetch origin 5b734fb7edfdf3f8a836a3ddee81eba506230fdd
git reset --hard 5b734fb7edfdf3f8a836a3ddee81eba506230fdd
git apply --ignore-whitespace /triton/vllm-neuron-issue-1.patch

mkdir -p $LOG_ROOT
OUTPUT_LOG="$LOG_ROOT/triton_server.log"
rm -rf $MODEL_REPO
mkdir -p $MODEL_REPO
VERSION=1
MODEL_NAME=$MODEL_NAME
mkdir -p $MODEL_REPO/$MODEL_NAME/$VERSION
cp /triton/model.json $MODEL_REPO/$MODEL_NAME/$VERSION/model.json
cp /triton/config.pbtxt $MODEL_REPO/$MODEL_NAME/config.pbtxt
cd $GIT_CLONE_DIR
pip3 install -r requirements-neuron.txt
pip3 install .
pip3 install triton==2.2.0
pip3 install pynvml==11.5.3
git clone https://github.com/triton-inference-server/vllm_backend.git /tmp/vllm_backend
cd /tmp/vllm_backend
git fetch origin 507e4dccabf85c3b7821843261bcea7ea5828802
git reset --hard 507e4dccabf85c3b7821843261bcea7ea5828802

mkdir -p /opt/tritonserver/backends/vllm
cp -r /tmp/vllm_backend/src/* /opt/tritonserver/backends/vllm/
cd $GIT_CLONE_DIR
export NEURON_CC_FLAGS="--model-type transformer"
export NEURON_COMPILE_CACHE_URL="$CACHE_DIR"
export OMP_NUM_THREADS=$OPM_NUM_THREADS
tritonserver \
--model-repository=${MODEL_REPO} \
--http-port=$HTTP_PORT \
--disable-auto-complete-config \
--log-file=$OUTPUT_LOG \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"