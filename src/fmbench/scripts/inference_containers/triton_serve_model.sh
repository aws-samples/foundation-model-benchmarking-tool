HF_MODEL=/$1
UNIFIED_CKPT_PATH=/tmp/ckpt/$HF_MODEL
ENGINE_DIR=/engines
CONVERT_CHKPT_SCRIPT=/tensorrtllm_backend/tensorrt_llm/examples/llama/convert_checkpoint.py
TP_DEGREE=$2
WORKERS=$TP_DEGREE
BATCH_SIZE=$3
WORLD_SIZE=$TP_DEGREE
MODEL_COPIES=$4
HTTP_PORT=$5
GRPC_PORT=$((HTTP_PORT + 1))
METRICS_PORT=$((HTTP_PORT + 2))


echo HF_MODEL=$HF_MODEL, UNIFIED_CKPT_PATH=$UNIFIED_CKPT_PATH, ENGINE_DIR=$ENGINE_DIR, CONVERT_CHKPT_SCRIPT=$CONVERT_CHKPT_SCRIPT
echo TP_DEGREE=$TP_DEGREE, WORKERS=$WORKERS, BATCH_SIZE=$BATCH_SIZE, WORLD_SIZE=$WORLD_SIZE, HTTP_PORT=$HTTP_PORT, GRPC_PORT=$GRPC_PORT, METRICS_PORT=$METRICS_PORT

cmd="python3 ${CONVERT_CHKPT_SCRIPT} --model_dir ${HF_MODEL} --output_dir ${UNIFIED_CKPT_PATH} --dtype float16 --tp_size $TP_DEGREE --workers $WORKERS"
echo going to run $cmd
$cmd

cmd="trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} --remove_input_padding enable --gpt_attention_plugin float16 --context_fmha enable --gemm_plugin float16 --output_dir ${ENGINE_DIR} --max_batch_size $BATCH_SIZE --workers $WORKERS"
echo going to run $cmd
$cmd

inflight_batcher_llm="/tensorrtllm_backend/all_models/inflight_batcher_llm"
TRITON_SERVER_DIR="/opt/tritonserver/."
echo going to copy $inflight_batcher_llm to $inflight_batcher_llm
cp -R $inflight_batcher_llm $TRITON_SERVER_DIR

# preprocessing
TOKENIZER_DIR=$HF_MODEL/
TOKENIZER_TYPE=auto
ENGINE_DIR=/engines
DECOUPLED_MODE=false
MODEL_FOLDER=/opt/tritonserver/inflight_batcher_llm
MAX_BATCH_SIZE=$BATCH_SIZE
INSTANCE_COUNT=1
MAX_QUEUE_DELAY_MS=10000
TRITON_BACKEND=tensorrtllm
FILL_TEMPLATE_SCRIPT=/tensorrtllm_backend/tools/fill_template.py
echo "going to run a whole bunch of pre and post processing commands"
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},tokenizer_type:${TOKENIZER_TYPE},triton_max_batch_size:${MAX_BATCH_SIZE},preprocessing_instance_count:${INSTANCE_COUNT}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},tokenizer_type:${TOKENIZER_TYPE},triton_max_batch_size:${MAX_BATCH_SIZE},postprocessing_instance_count:${INSTANCE_COUNT}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},bls_instance_count:${INSTANCE_COUNT}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/ensemble/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm/config.pbtxt triton_backend:${TRITON_BACKEND},triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},batching_strategy:inflight_fused_batching,exclude_input_in_output:True
# from this blog https://developer.nvidia.com/blog/turbocharging-meta-llama-3-performance-with-nvidia-tensorrt-llm-and-nvidia-triton-inference-server/
#python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm/config.pbtxt triton_backend:${TRITON_BACKEND},triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},batching_strategy:inflight_fused_batching,max_beam_width:1,max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False

echo "after running a whole bunch of pre and post processing commands"

# start triton servers
for ((i=0; i<$MODEL_COPIES; i++)); do
  # Calculate the start and end GPU IDs based on tp_degree
  start_id=$((i * TP_DEGREE))
  end_id=$((start_id + TP_DEGREE - 1))
  
  # Create the CUDA_VISIBLE_DEVICES string as a comma-separated list
  cvd=$(seq -s, $start_id $end_id)

  # Adjust port numbers for each iteration (increment by 3)
  http_port=$((HTTP_PORT + i * 3))
  grpc_port=$((GRPC_PORT + i * 3))
  metrics_port=$((METRICS_PORT + i * 3))

  echo "Starting server $((i + 1))"
  echo "CUDA_VISIBLE_DEVICES=$cvd"

  # Construct the command with updated CUDA_VISIBLE_DEVICES and port numbers
  export CUDA_VISIBLE_DEVICES=$cvd
  cmd="python3 /tensorrtllm_backend/scripts/launch_triton_server.py \
    --world_size=$WORLD_SIZE --model_repo=/opt/tritonserver/inflight_batcher_llm \
    --http_port $http_port --grpc_port $grpc_port --metrics_port $metrics_port"
  
  echo "Going to start the Triton server with the following command: $cmd"
  
  # Execute the command
  $cmd

  # wait for a few seconds so that the previous instance is able to bind to the GPUs
  sleep 10
done
