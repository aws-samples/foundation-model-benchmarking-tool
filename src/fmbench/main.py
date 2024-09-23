import os
import re
import sys
import yaml
import json
import time
import boto3
import logging
import argparse
import requests
import papermill as pm
from typing import Dict
from pathlib import Path
from datetime import datetime
from nbformat import NotebookNode

# Setup logging
logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Create an S3 client
s3_client = boto3.client('s3')

# Function to check whether the given uri is a valid s3 uri
def is_valid_s3_uri(s3_uri: str) -> bool:
    pattern = re.compile(r's3://[^/]+/.+')
    return bool(pattern.match(s3_uri))

# Function to check whether the given uri is a valid https URL
def is_valid_http_url(url: str) -> bool:
    return url.startswith("https://") or url.startswith("http://")

# Function to get the s3_uri from the user and get the config file path, writing the txt file
def read_config(config_file_path: str) -> Dict:
    if is_valid_s3_uri(config_file_path):
        logger.info(f"executing the config file found in {config_file_path}...")

        bucket, key = config_file_path.replace("s3://", "").split("/", 1)

        # Get object from S3 and load YAML
        response = s3_client.get_object(Bucket=bucket, Key=key)
        config_content = yaml.safe_load(response["Body"])
    elif is_valid_http_url(config_file_path):
        try:
            logger.info(f"loading config from HTTPS URL: {config_file_path}")
            response = requests.get(config_file_path)
            response.raise_for_status()  # Raises a HTTPError for bad responses
            config_content = yaml.safe_load(response.text)
        except requests.exceptions.RequestException as e:
            logger.error(f"error loading config from HTTPS URL: {e}")
            raise
    else:
        logger.info(f"the provided URI '{config_file_path}' is not a valid S3 URI or HTTPS URL, assuming this is a local file")
        config_content = yaml.safe_load(Path(config_file_path).read_text())

    # You can choose to write to config_path here if needed, otherwise just return the loaded content
    logger.info(f"loaded configuration: {json.dumps(config_content, indent=2)}")
    return config_content

# Function to handle cell outputs
def output_handler(cell: NotebookNode, _):
    if cell.cell_type == 'code':
        for output in cell.get('outputs', []):
            if output.output_type == 'stream':
                print(output.text, end='')


def run_notebooks(config_file: str) -> None:
    # Assume `read_config` function is defined elsewhere to load the config
    config = read_config(config_file)

    current_directory = Path(__file__).parent
    logging.info(f"Current directory is --> {current_directory}")

    output_directory = current_directory / "executed_notebooks"
    if not output_directory.exists():
        output_directory.mkdir()

    all_notebooks = list(current_directory.glob('*.ipynb'))
    logger.info(f"Steps to be executed: {all_notebooks}")

    for step, execute in config['run_steps'].items():
        if execute:
            notebook_path = current_directory / step

            if not notebook_path.exists():
                matching_notebooks = [nb for nb in all_notebooks if nb.name[1:] == step[1:]]
                if matching_notebooks:
                    notebook_path = matching_notebooks[0]
                else:
                    logging.error(f"No matching notebook found for step: {step}")
                    continue

            logging.info(f"Current step file --> {notebook_path.stem}")

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_file = output_directory / f"{notebook_path.stem}_{timestamp}.ipynb"

            try:
                logging.info(f"Executing {notebook_path.name}...")
                logger.info(f"THE STEP BEING EXECUTED NOW: {step}")
                pm.execute_notebook(
                    input_path=str(notebook_path),
                    output_path=str(output_file),
                    kernel_name='python3',
                    parameters={},
                    report_mode=True,  
                    progress_bar=True,
                    stdout_file=None, 
                    stderr_file=None,
                    log_output=True,
                    output_handler=output_handler 
                )
                logger.info(f"STEP EXECUTION COMPLETED: {step}")
            except FileNotFoundError as e:
                logging.error(f"File not found: {e.filename}")
                sys.exit(1)
            except Exception as e:
                logging.error(f"Failed to execute {step}: {str(e)}")
                sys.exit(1)
        else:
            logging.info(f"Skipping {step} as it is not marked for execution")

    logger.info(f"FMBench has completed the benchmarking process. Check the \"results-*\" folder for results")


# main function to run all of the fmbench process through a single command via this python package
def main():
    parser = argparse.ArgumentParser(description='Run FMBench with a specified config file.')
    parser.add_argument('--config-file', type=str, help='The S3 URI of your Config File', required=True)
    role_help = 'The ARN of the role to be used for FMBench. If an \
                 Amazon SageMaker endpoint is being deployed \
                 through FMBench then this role would also be used \
                 by that endpoint'
    parser.add_argument('--role-arn', type=str, default=None, required=False, help=role_help)
    # add an option to run FMBench local mode. If local mode is set to yes, then FMBench uses read and write data locally and if no, 
    # then the test will continue to interact with S3
    parser.add_argument('--local-mode', type=str, default=None, choices=['yes', 'no'], help='Specify if running in local mode or not. Options: yes, no. Default is no.')
    # add an option to run FMBench with a custom tmp file argument. Users if running in local mode can configure a custom tmp file 
    # instead of using the default /tmp directory
    parser.add_argument('--tmp-dir', type=str, default=None, required=False, help='An optional tmp directory if fmbench is running in local mode.')
    parser.add_argument('--write-bucket', type=str, help='Write bucket that is used for sagemaker endpoints in local mode and storing metrics in s3 mode.')

    args = parser.parse_args()
    print(f"main, {args} = args")

    # Set the environment variable based on the parsed argument
    os.environ["CONFIG_FILE_FMBENCH"] = args.config_file
    logger.info(f"Config file specified: {args.config_file}")
    
    # set env var to indicate that fmbench is being run from main and not interactively via a notebook
    os.environ["INTERACTIVE_MODE_SET"] = "no"

    # set the environment variable for the local mode option
    if args.local_mode:
        print(f"setting the LOCAL_MODE to {args.local_mode}")
        os.environ["LOCAL_MODE"] = args.local_mode
        if args.local_mode == "yes":
            if args.write_bucket is None:
                logger.error("Write bucket is not provided when local mode is set to 'yes'")
                sys.exit(1)
            else:
                # set the environment variable for the write bucket name to be configured and used for sagemaker endpoints and 
                # metrics stored in s3 mode with local mode is set to "yes" by the user
                os.environ["WRITE_BUCKET"] = args.write_bucket
                logger.info(f"Write bucket specified in local mode: {args.write_bucket}")
            if args.tmp_dir:
                os.environ["TMP_DIR"] = args.tmp_dir
                logger.info(f"tmp directory specified in local mode: {args.tmp_dir}")
            else:
                logger.info(f"Custom tmp file not provided.")


    # if a role arn is specified then set it as an env var so that the rest of the code
    # can use it. This will then be used to set the templatized "sagemaker_execution_role"
    # parameter in the config file
    if args.role_arn:
        print(f"setting FMBENCH_ROLE_ARN env variable to {args.role_arn}")
        os.environ['FMBENCH_ROLE_ARN'] = args.role_arn

    # Proceed with the rest of your script's logic, passing the config file as needed
    run_notebooks(args.config_file)    

if __name__ == "__main__":
    main()

