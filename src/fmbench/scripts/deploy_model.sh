# script for deploying model in EC2
# Download the model from the container and deploy
echo "going to download model now"
echo "content nin docket command: $REGION, $IMAGE_URI, $MODEL_NAME, $HF_TOKEN"
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $IMAGE_URI

docker pull $IMAGE_URI

docker run -it --runtime=nvidia --gpus all --shm-size 12g \
 -v ~/home/ubuntu/$MODEL_NAME:/opt/ml/model:ro \
 -v /home/ubuntu/model_server_logs:/opt/djl/logs \
 -e HF_TOKEN=$HF_TOKEN \
 -p 8080:8080 \
 $IMAGE_URI\

 echo "done pulling model"
