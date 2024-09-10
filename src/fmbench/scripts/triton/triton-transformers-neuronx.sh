#!/bin/bash
[ ! -d /triton ] && echo "/triton dir must exist" && exit 1
[ $# -ne 3 ] && echo "usage: $0 hf-model-id model-name http-port" && exit 1

# The HF model id, model name and http port
# are parsed from the triton multi model preparation
# function here
HF_MODEL_ID=$1
MODEL_NAME=$2
HTTP_PORT=$3

LOG_ROOT=/triton/logs
MODEL_REPO=/triton/model_repository
CACHE_DIR=/cache

mkdir -p $LOG_ROOT
OUTPUT_LOG="$LOG_ROOT/triton-server.log"
rm -rf $MODEL_REPO
mkdir -p $MODEL_REPO
VERSION=1
mkdir -p $MODEL_REPO/$MODEL_NAME/$VERSION

# Copy the model repository files into the triton container
cp /triton/model.py $MODEL_REPO/$MODEL_NAME/$VERSION/model.py
cp /triton/model.json $MODEL_REPO/$MODEL_NAME/$VERSION/model.json
cp /triton/config.pbtxt $MODEL_REPO/$MODEL_NAME/config.pbtxt

export NEURON_CC_FLAGS="--model-type transformer"
export NEURON_COMPILE_CACHE_URL="$CACHE_DIR"
tritonserver \
--model-repository=${MODEL_REPO} \
--http-port=$HTTP_PORT \
--disable-auto-complete-config \
--log-file=$OUTPUT_LOG \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"
