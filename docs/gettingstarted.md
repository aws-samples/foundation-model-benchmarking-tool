# Getting started with `FMBench`

`FMBench` is available as a Python package on [PyPi](https://pypi.org/project/fmbench) and is run as a command line tool once it is installed. All data that includes metrics, reports and results are stored in an Amazon S3 bucket.

While technically you can run `FMBench` on any AWS compute but practically speaking we either run it on a SageMaker Notebook or on EC2. Both these options are described below.

👉 The following sections are discussing running `FMBench` the tool, as different from where the FM is actually deployed. For example, we could run `FMBench` on EC2 but the model being deployed is on SageMaker or even Bedrock.
