import os
import re
import yaml
import json
import boto3
import logging
import nbformat
import argparse
import requests
import papermill as pm
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
def is_valid_https_url(url: str) -> bool:
    return url.startswith("https://")

# Function to get the s3_uri from the user and get the config file path, writing the txt file
def write_config_file_path(user_input: str) -> dict:
    current_working_directory = Path.cwd()
    config_path = current_working_directory / 'fmbt' / 'config_filepath.txt'
    
    if is_valid_s3_uri(user_input):
        logger.info(f"Executing the config file found in {user_input}...")

        bucket, key = user_input.replace("s3://", "").split("/", 1)

        # Get object from S3 and load YAML
        response = s3_client.get_object(Bucket=bucket, Key=key)
        config_content = yaml.safe_load(response["Body"])
    elif is_valid_https_url(user_input):
        try:
            logger.info(f"Loading config from HTTPS URL: {user_input}")
            response = requests.get(user_input)
            response.raise_for_status()  # Raises a HTTPError for bad responses
            config_content = yaml.safe_load(response.text)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error loading config from HTTPS URL: {e}")
            raise
    else:
        raise ValueError(f"The provided URI '{user_input}' is not a valid S3 URI or HTTPS URL.")
    
    # You can choose to write to config_path here if needed, otherwise just return the loaded content
    print(f"Loaded configuration: {config_content}")
    return config_content

# Function to handle cell outputs
def output_handler(cell: NotebookNode, _):
    if cell.cell_type == 'code':
        for output in cell.get('outputs', []):
            if output.output_type == 'stream':
                print(output.text, end='')


def run_notebooks(config_file):
    # Assume `write_config_file_path` function is defined elsewhere to load the config
    config = write_config_file_path(config_file)

    current_directory = Path(__file__).parent
    logging.info(f"Current directory is --> {current_directory}")

    output_directory = current_directory / "executed_notebooks"
    if not output_directory.exists():
        output_directory.mkdir()

    for step, execute in config['run_steps'].items():
        if execute:
            notebook_path = current_directory / step
            logging.info(f"Current step file --> {notebook_path.stem}")

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_file = output_directory / f"{notebook_path.stem}_{timestamp}.ipynb"

            try:
                logging.info(f"Executing {notebook_path.name}...")
                logger.info(f"THE STEP BEING EXECUTED NOW: {step}")
                pm.execute_notebook(
                    input_path=str(notebook_path),
                    output_path=str(output_file),
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
            except Exception as e:
                logging.error(f"Failed to execute {step}: {str(e)}")
        else:
            logging.info(f"Skipping {step} as it is not marked for execution.")

    print("FMBT has completed the benchmarking process. Check your S3 bucket for results.")


# main function to run all of the fmbt process through a single command via this python package
def main():
    parser = argparse.ArgumentParser(description='Run FMBT with a specified config file.')
    parser.add_argument('--config-file', type=str, help='The S3 URI of your Config File', required=True)
    args = parser.parse_args()
    print(f"{args} = args")

    # Set the environment variable based on the parsed argument
    os.environ["CONFIG_FILE_FMBT"] = args.config_file
    logger.info(f"Config file specified: {args.config_file}")

    # Proceed with the rest of your script's logic, passing the config file as needed
    run_notebooks(args.config_file)
    logger.info(f"Environment variable CONFIG_FILE_FMBT set to: {os.environ['CONFIG_FILE_FMBT']}")

if __name__ == "__main__":
    main()

