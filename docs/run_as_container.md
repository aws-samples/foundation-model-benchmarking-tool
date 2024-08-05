# Run `FMBench` as a Docker container

You can now run `FMBench` on any platform where you can run a Docker container, for example on an EC2 VM, SageMaker Notebook etc. The advantage is that you do not have to install anything locally, so no `conda` installs needed anymore. Here are the steps to do that.

1. Create local directory structure needed for `FMBench` and copy all publicly available dependencies from the AWS S3 bucket for `FMBench`. This is done by running the `copy_s3_content.sh` script available as part of the `FMBench` repo. You can place model specific tokenizers and any new configuration files you create in the `/tmp/fmbench-read` directory that is created after running the following command. 

    ```{.bash}
    curl -s https://raw.githubusercontent.com/aws-samples/foundation-model-benchmarking-tool/main/copy_s3_content.sh | sh
    ```

1. That's it! You are now ready to run the container.

    ```{.bash}
    # set the config file path to point to the config file of interest
    CONFIG_FILE=https://raw.githubusercontent.com/aws-samples/foundation-model-benchmarking-tool/main/src/fmbench/configs/llama2/7b/config-llama2-7b-g5-quick.yml
    docker run -v $(pwd)/fmbench:/app \
      -v /tmp/fmbench-read:/tmp/fmbench-read \
      -v /tmp/fmbench-write:/tmp/fmbench-write \
      aarora79/fmbench:v1.0.47 \
     "fmbench --config-file ${CONFIG_FILE} --local-mode yes --write-bucket placeholder > fmbench.log 2>&1"
    ```
    
1. The above command will create a `fmbench` directory inside the current working directory. This directory contains the `fmbench.log` and the `results-*` folder that is created once the run finished.
