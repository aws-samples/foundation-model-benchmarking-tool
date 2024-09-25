    #!/usr/bin/env bash

    # This script builds a Docker image and saves it locally in the home directory.

    # Set the image name and tag
    export IMAGE_NAME=tritonserver-neuronx
    export IMAGE_TAG=fmbench

    # Get the directory of the current script
    DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

    # Build the Docker image locally with the image name
    docker build -f ${DIR}/Dockerfile_triton -t ${IMAGE_NAME}:${IMAGE_TAG} ${DIR}/..

    if [ $? -ne 0 ]; then
        echo "Error: Docker image build failed"
        exit 1
    fi

    # Save the Docker image locally in the home directory
    HOME_DIR=$(eval echo ~$USER)
    docker save ${IMAGE_NAME}:${IMAGE_TAG} | gzip > ${HOME_DIR}/${IMAGE_NAME}_${IMAGE_TAG}.tar.gz

    if [ $? -ne 0 ]; then
        echo "Error: Failed to save Docker image"
        exit 1
    else
        echo "Triton docker image saved as ${IMAGE_NAME}_${IMAGE_TAG}.tar.gz in the home directory (${HOME_DIR})."
    fi
