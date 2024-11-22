#!/bin/bash
# this scripts creates a local directory for running FMBench
# without any s3 dependency and copies all relevant files
# for public FMBench content
BASE_DIR=${1:-/tmp}
FMBENCH_READ_DIR=$BASE_DIR/fmbench-read
FMBENCH_WRITE_DIR=$BASE_DIR/fmbench-write
BUCKET=aws-blogs-artifacts-public

mkdir -p $FMBENCH_READ_DIR
mkdir -p $FMBENCH_READ_DIR/tokenizer
mkdir -p $FMBENCH_READ_DIR/llama2_tokenizer
mkdir -p $FMBENCH_READ_DIR/llama3_tokenizer
mkdir -p $FMBENCH_READ_DIR/llama3_1_tokenizer
mkdir -p $FMBENCH_READ_DIR/llama3_2_tokenizer
mkdir -p $FMBENCH_READ_DIR/mistral_tokenizer
wget https://${BUCKET}.s3.amazonaws.com/artifacts/ML-FMBT/manifest.txt -P ${FMBENCH_READ_DIR}/

# copy each file of the public content for FMBench
for i in `cat ${FMBENCH_READ_DIR}/manifest.txt`
do
  # Skip if filename contains ".keep" in it
  if echo "$i" | grep -q ".keep"; then
    continue
  fi
  dir_path=`dirname $i`
  mkdir -p ${FMBENCH_READ_DIR}/$dir_path
  wget https://${BUCKET}.s3.amazonaws.com/artifacts/ML-FMBT/$i -P ${FMBENCH_READ_DIR}/$dir_path
done
