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
import subprocess  # Added for executing .py files
from typing import Dict
from pathlib import Path
from datetime import datetime

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

# Adjusting this function to handle .py files as steps to each portion of fmbench
def run_scripts(config_file: str) -> None:
    config = read_config(config_file)

    current_directory = Path(__file__).parent
    logging.info(f"Current directory is --> {current_directory}")

    output_directory = current_directory / "executed_scripts"
    if not output_directory.exists():
        output_directory.mkdir()

    for step, execute in config['run_steps'].items():
        if execute:
            script_path = current_directory / f"{step}.py"  # Adjusted for .py file
            logging.info(f"Current step file --> {script_path.stem}")

            try:
                logging.info(f"Executing {script_path.name}...")
                logger.info(f"THE STEP BEING EXECUTED NOW: {step}")
                # Using subprocess to run the .py file and capture output
                result = subprocess.run(
                    ["python", str(script_path)], capture_output=True, text=True, check=True
                )
                logger.info(f"Output: {result.stdout}")
                logger.info(f"Error (if any): {result.stderr}")
                logger.info(f"STEP EXECUTION COMPLETED: {step}")
            except FileNotFoundError as e:
                logging.error(f"File not found: {e.filename}")
                sys.exit(1)
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to execute {step}: {e.stderr}")
                sys.exit(1)
        else:
            logging.info(f"Skipping {step} as it is not marked for execution")

    logger.info(f"FMBench has completed the benchmarking process. Check the output for results")

# Adjust the main function as necessary
def main():
    parser = argparse.ArgumentParser(description='Run FMBench with a specified config file.')
    parser.add_argument('--config-file', type=str, help='The S3 URI or local path of your Config File', required=True)
    args = parser.parse_args()
    print(f"{args} = args")

    os.environ["CONFIG_FILE_FMBENCH"] = args.config_file
    logger.info(f"Config file specified: {args.config_file}")

    run_scripts(args.config_file)  # Call the updated function

if __name__ == "__main__":
    main()