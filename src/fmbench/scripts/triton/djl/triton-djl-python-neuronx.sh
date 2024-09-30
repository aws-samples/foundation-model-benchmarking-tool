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

pip3 install --extra-index-url https://pip.repos.neuron.amazonaws.com optimum[neuronx]
GIT_CLONE_DIR=/tmp/djl-serving
git clone https://github.com/deepjavalibrary/djl-serving.git $GIT_CLONE_DIR
cd $GIT_CLONE_DIR
git fetch origin c343d60b35f0d42f96f678570a553953f055ab32
git reset --hard c343d60b35f0d42f96f678570a553953f055ab32
cd $GIT_CLONE_DIR/engines/python/setup 
pip3 install .

GIT_CLONE_DIR=/tmp/vllm
git clone https://github.com/vllm-project/vllm.git $GIT_CLONE_DIR
cd $GIT_CLONE_DIR
git fetch origin 38c4b7e863570a045308af814c72f4504297222e
git reset --hard 38c4b7e863570a045308af814c72f4504297222e
pip3 install -r requirements-neuron.txt
pip3 install .
pip3 install pynvml==11.5.3 transformers==4.44.2

mkdir -p $LOG_ROOT
OUTPUT_LOG="$LOG_ROOT/triton-server.log"
rm -rf $MODEL_REPO
mkdir -p $MODEL_REPO
VERSION=1
MODEL_NAME=$MODEL_NAME
mkdir -p $MODEL_REPO/$MODEL_NAME/$VERSION
cp /triton/model.py $MODEL_REPO/$MODEL_NAME/$VERSION/model.py
cp /triton/model.json $MODEL_REPO/$MODEL_NAME/$VERSION/model.json
cp /triton/config.pbtxt $MODEL_REPO/$MODEL_NAME/config.pbtxt
export NEURON_CC_FLAGS="--model-type transformer"
export NEURON_COMPILE_CACHE_URL="$CACHE_DIR"
export OMP_NUM_THREADS=$OPM_NUM_THREADS
tritonserver \
--model-repository=${MODEL_REPO} \
--http-port=$HTTP_PORT \
--disable-auto-complete-config \
--log-file=$OUTPUT_LOG \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"