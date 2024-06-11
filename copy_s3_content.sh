#!/bin/bash
set -e

# this scripts creates a local directory for running FMBench
# without any s3 dependency and copies all relevant files
# for public FMBench content
FMBENCH_READ_DIR=/tmp/fmbench-read
FMBENCH_WRITE_DIR=/tmp/fmbench-write
BUCKET=aws-blogs-artifacts-public

mkdir -p $FMBENCH_WRITE_DIR
mkdir -p $FMBENCH_READ_DIR/tokenizer
mkdir -p $FMBENCH_READ_DIR/llama2_tokenizer
mkdir -p $FMBENCH_READ_DIR/llama3_tokenizer
mkdir -p $FMBENCH_READ_DIR/mistral_tokenizer
curl --output-dir ${FMBENCH_READ_DIR}/ -O https://${BUCKET}.s3.amazonaws.com/artifacts/ML-FMBT/manifest.txt

# copy each file of the public content for FMBench
for i in `cat ${FMBENCH_READ_DIR}/manifest.txt`
do
  dir_path=`dirname $i`
  mkdir -p ${FMBENCH_READ_DIR}/$dir_path
  curl --output-dir ${FMBENCH_READ_DIR}/$dir_path -O https://${BUCKET}.s3.amazonaws.com/artifacts/ML-FMBT/$i
done
