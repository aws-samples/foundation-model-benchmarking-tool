# Run `FMBench` on Amazon EC2

For some enterprise scenarios it might be desirable to run `FMBench` directly on an EC2 instance with no dependency on S3. Here are the steps to do this:

1. Have a `t3.xlarge` (or larger) instance in the `Running` stage. Make sure that the instance has at least 50GB of disk space and the IAM role associated with your EC2 instance has `AmazonSageMakerFullAccess` policy associated with it and `sagemaker.amazonaws.com` added to its Trust relationships.
    ```{.bash}
    {
        "Effect": "Allow",
        "Principal": {
            "Service": "sagemaker.amazonaws.com"
        },
        "Action": "sts:AssumeRole"
    }
    ```

1. Setup the `fmbench_python311` conda environment. This step required conda to be installed on the EC2 instance, see [instructions](https://www.anaconda.com/download) for downloading Anaconda.

    ```{.bash}
    conda create --name fmbench_python311 -y python=3.11 ipykernel
    source activate fmbench_python311;
    pip install -U fmbench
    ```

1. Create local directory structure needed for `FMBench` and copy all publicly available dependencies from the AWS S3 bucket for `FMBench`. This is done by running the `copy_s3_content.sh` script available as part of the `FMBench` repo. **Replace `/tmp` in the command below with a different path if you want to store the config files and the `FMBench` generated data in a different directory**.

    ```{.bash}
    # Replace "/tmp" with "/path/to/your/custom/tmp" if you want to use a custom tmp directory
    TMP_DIR="/tmp"
    curl -s https://raw.githubusercontent.com/aws-samples/foundation-model-benchmarking-tool/main/copy_s3_content.sh | sh -s -- "$TMP_DIR"
    ```


1. Run `FMBench` with a packaged or a custom config file. The `--write-bucket` parameter value is just a placeholder and an actual S3 bucket is not required. You could set the `--tmp-dir` flag to an EFA path instead of `/tmp` if using a shared path for storing config files and reports.

    ```{.bash}
    fmbench --config-file $TMP_DIR/fmbench-read/configs/llama3/8b/config-ec2-llama3-8b.yml --local-mode yes --write-bucket placeholder --tmp-dir $TMP_DIR > fmbench.log 2>&1
    ```

1. Open a new Terminal and navigate to the `foundation-model-benchmarking-tool` directory and do a `tail` on `fmbench.log` to see a live log of the run.

    ```{.bash}
    tail -f fmbench.log
    ```

1. All metrics are stored in the `/tmp/fmbench-write` directory created automatically by the `fmbench` package. Once the run completes all files are copied locally in a `results-*` folder as usual.
