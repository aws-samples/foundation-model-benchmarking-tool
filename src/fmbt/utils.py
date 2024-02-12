import re
import os
import yaml
import boto3
import logging
import requests
import posixpath
import unicodedata
from typing import Dict
from pathlib import Path
from fmbt import globals
from transformers import AutoTokenizer
from botocore.exceptions import NoCredentialsError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import math
class CustomTokenizer:    
    """A custom tokenizer class"""
    TOKENS: int = 1000
    WORDS: int = 750

    def __init__(self, bucket, prefix, local_dir):
        print(f"CustomTokenizer, based on HF transformers")
        # Check if the tokenizer files exist locally
        if not os.path.exists(local_dir):
            # If not, download from S3            
            _download_from_s3(bucket, prefix, local_dir)
        # Load the tokenizer from the local directory
        dir_not_empty = any(Path(local_dir).iterdir())
        if dir_not_empty is True:
            logger.info("loading the provided tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
        else:
            logger.error(f"no tokenizer provided, the {local_dir} is empty, "
                         f"using default tokenizer i.e. {self.WORDS} words = {self.TOKENS} tokens")
            self.tokenizer = None

    def count_tokens(self, text):
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text))
        else:
            int(math.ceil((self.TOKENS/self.WORDS) * len(text.split())))
    
_tokenizer = CustomTokenizer(globals.READ_BUCKET_NAME, globals.TOKENIZER_DIR_S3, globals.TOKENIZER)

# utility functions
def load_config(config_file) -> Dict:
    """
    Load configuration from a local file or an S3 URI.

    :param config_file: Path to the local file or S3 URI (s3://bucket/key)
    :return: Dictionary with the loaded configuration
    """

    # Check if config_file is an S3 URI
    if config_file.startswith("s3://"):
        try:
            # Parse S3 URI
            s3_client = boto3.client('s3')
            bucket, key = config_file.replace("s3://", "").split("/", 1)

            # Get object from S3 and load YAML
            response = s3_client.get_object(Bucket=bucket, Key=key)
            return yaml.safe_load(response["Body"])
        except NoCredentialsError:
            print("AWS credentials not found.")
            raise
        except Exception as e:
            print(f"Error loading config from S3: {e}")
            raise
    # Check if config_file is an HTTPS URL
    elif config_file.startswith("https://"):
        try:
            response = requests.get(config_file)
            response.raise_for_status()  # Raises a HTTPError if the response was an error
            return yaml.safe_load(response.text)
        except requests.exceptions.RequestException as e:
            print(f"Error loading config from HTTPS URL: {e}")
            raise
    else:
        # Assume local file system if not S3 or HTTPS
        try:
            with open(config_file, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading config from local file system: {e}")
            raise
   
# The files in LongBench contain nonstandard or irregular Unicode.
# For compatibility and safety we normalize them.
def _normalize(text, form='NFC'):
    return unicodedata.normalize(form, text)

def _download_from_s3(bucket_name, prefix, local_dir):
    """Downloads files from an S3 bucket and a specified prefix to a local directory."""
    s3_client = boto3.client('s3')

    # Ensure the local directory exists
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # List and download files
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' in response:
            for obj in response['Contents']:
                file_key = obj['Key']
                if file_key.endswith('/'):  
                    continue
                local_file_path = os.path.join(local_dir, os.path.basename(file_key))
                s3_client.download_file(bucket_name, file_key, local_file_path)
                logger.info(f"Downloaded: {local_file_path}")
        else:
            logger.warning(f"No files found in S3 Bucket: '{bucket_name}' with Prefix: '{prefix}'")
    except Exception as e:
        logger.error(f"An error occurred while downloading from S3: {e}")


def count_tokens(text: str) -> int:
    global _tokenizer
    return len(_tokenizer.count_tokens(text))

def process_item(item, prompt_fmt: str) -> Dict:
    question = _normalize(item.input)
    context = _normalize(item.context)
    prompt = prompt_fmt.format(question=question, context=context)
    prompt_len = count_tokens(prompt)
    ## generalize this further...
    ## bring your own script (if different) - bring your count token and your script
    return {
        "question": question,
        "context": context,
        "prompt": prompt,
        "prompt_len": prompt_len,
        "question_len": len(_tokenizer.count_tokens(question)),
        "context_len": len(_tokenizer.count_tokens(context)),
    }

def nt_to_posix(p: str) -> str:
    return p.replace("\\", "/")

# Function to write data to S3
def write_to_s3(json_data, bucket_name, dir1, dir2, file_name):

    # Initialize S3 client
    s3_client = boto3.client('s3')

    # Construct the S3 file path
    s3_file_path = posixpath.join(nt_to_posix(dir1), nt_to_posix(dir2), file_name)
    logger.info(f"write_to_s3, s3_file_path={s3_file_path}")
    try:
        # Write the JSON data to the S3 bucket
        s3_client.put_object(Bucket=bucket_name, Key=s3_file_path, Body=json_data)
        return (f"s3://{bucket_name}/{s3_file_path}")
    except NoCredentialsError:
        logger.error("write_to_s3, Error: AWS credentials not found.")
    except Exception as e:
        logger.error(f"write_to_s3, An error occurred: {e}")

        
## function to read from s3
def read_from_s3(bucket_name, s3_file_path):

    # Initialize S3 client
    s3_client = boto3.client('s3')
    s3_file_path = nt_to_posix(s3_file_path)

    try:
        # Fetch the object from S3
        logger.error(f"read_from_s3, reading file from bucket={bucket_name}, key={s3_file_path}")
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_file_path)
        
        return response['Body'].read().decode('utf-8')
    except NoCredentialsError:
        logger.error("read_from_s3, Error: AWS credentials not found.")
        return None
    except Exception as e:
        logger.error(f"read_from_s3, An error occurred: {e}")
        return None

## gets a single s3 file
def get_s3_object(bucket: str, key: str) -> str:
 
    key = nt_to_posix(key)
    logger.info(f"get_s3_object, bucket_name={bucket}, key={key}")

    # Create an S3 client
    s3_client = boto3.client('s3')

    # Retrieve the object from S3
    response = s3_client.get_object(Bucket=bucket, Key=key)

    # Read the content of the file
    content = response['Body'].read().decode('utf-8')

    return content

# Function to list files in S3 bucket with a specific prefix
def list_s3_files(bucket, prefix, suffix='.json'):
    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=nt_to_posix(prefix))
    return [item['Key'] for item in response.get('Contents', []) if item['Key'].endswith(suffix)]
