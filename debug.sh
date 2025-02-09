# script for a debug/developer workflow
# 1. Builds and install a local wheel
# 2. There is no step 2 :)

CONFIG_FILE_PATH=fmbench/configs/deepseek/config-deepseek-r1-quant1.58-longbench-byoe.yml
LOGFILE=fmbench.log

uv build
uv pip install -U dist/*.whl

# run the newly installed version
echo "going to run fmbench now"
fmbench --config-file $CONFIG_FILE_PATH  --local-mode yes --write-bucket placeholder --tmp-dir /tmp -A model_id=DeepSeek-R1-quant-1.58bit -A hf_tokenizer_model_id=deepseek-ai/DeepSeek-R1 -A instance_type=g6e.12xlarge -A results_dir=DeepSeek-R1-quant-1.58bit-g6e.12xl -A prompt_template=prompt_template_deepseek_longbench.txt  > $LOGFILE 2>&1

# Use FMBench to benchmark models on hosted on EC2 using the command below. If you want to write the metrics and results to an
# s3 bucket, replace `placeholder` with the name of that s3 bucket in your AWS account. Optionally, you can send the results to
# a custom tmp directory by setting the '--tmp-dir' argument followed by the path to that custom tmp directory. If '--tmp-dir' is not
# provided, the default 'tmp' directory will be used.
#fmbench --config-file $CONFIG_FILE_PATH --local-mode yes --write-bucket placeholder --tmp-dir /path/to/your_tmp_directory > $LOGFILE 2>&1
echo "all done"
