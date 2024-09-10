#!/bin/bash
[ ! -d /triton ] && echo "/triton dir must exist" && exit 1
[ $# -ne 2 ] && echo "usage: $0 hf-model-id model-name" && exit 1

HF_MODEL_ID=$1
MODEL_NAME=$2

LOG_ROOT=/triton/logs
MODEL_REPO=/triton/model_repository
CACHE_DIR=/cache

mkdir -p $LOG_ROOT
OUTPUT_LOG="$LOG_ROOT/triton-server.log"
rm -rf $MODEL_REPO
mkdir -p $MODEL_REPO
VERSION=1
mkdir -p $MODEL_REPO/$MODEL_NAME/$VERSION
# Set permissions for directories (read, write, and execute)
cp /triton/model.py $MODEL_REPO/$MODEL_NAME/$VERSION/model.py
cp /triton/model.json $MODEL_REPO/$MODEL_NAME/$VERSION/model.json
cp /triton/config.pbtxt $MODEL_REPO/$MODEL_NAME/config.pbtxt

export NEURON_CC_FLAGS="--model-type transformer"
export NEURON_COMPILE_CACHE_URL="$CACHE_DIR"
tritonserver \
--model-repository=${MODEL_REPO} \
--grpc-port=8001 \
--http-port=8000 \
--metrics-port=8002 \
--disable-auto-complete-config \
--log-file=$OUTPUT_LOG \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"