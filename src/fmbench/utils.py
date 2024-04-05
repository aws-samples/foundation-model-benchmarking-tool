import re
import os
import yaml
import math
import boto3
import logging
import requests
import posixpath
import unicodedata
from pathlib import Path
from fmbench import globals
from fmbench import defaults
from typing import Dict, List
from transformers import AutoTokenizer
from botocore.exceptions import NoCredentialsError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# The files in LongBench contain nonstandard or irregular Unicode.
# For compatibility and safety we normalize them.
def _normalize(text, form='NFC'):
    return unicodedata.normalize(form, str(text))

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
                logger.debug(f"Downloaded: {local_file_path}")
        else:
            logger.warning(f"No files found in S3 Bucket: '{bucket_name}' with Prefix: '{prefix}'")
    except Exception as e:
        logger.error(f"An error occurred while downloading from S3: {e}")

class CustomTokenizer:    
    """A custom tokenizer class"""
    TOKENS: int = 1000
    WORDS: int = 750

    def __init__(self, bucket, prefix, local_dir):
        print(f"CustomTokenizer, based on HF transformers")
        # Check if the tokenizer files exist in s3 and if not, use the autotokenizer       
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
            return int(math.ceil((self.TOKENS/self.WORDS) * len(text.split())))
    
_tokenizer = CustomTokenizer(globals.READ_BUCKET_NAME, globals.TOKENIZER_DIR_S3, globals.TOKENIZER)

# utility functions
def load_config(config_file) -> Dict:
    """
    Load configuration from a local file or an S3 URI.

    :param config_file: Path to the local file or S3 URI (s3://bucket/key)
    :return: Dictionary with the loaded configuration
    """
    session = boto3.session.Session()
    region_name = session.region_name
    caller = boto3.client('sts').get_caller_identity()
    account_id = caller.get('Account')
    arn_string = caller.get('Arn')

    # check if the file is still parameterized and if so replace the parameters with actual values
    # if the file is not parameterized then the following statements change nothing
    args = dict(region=session.region_name,
                role_arn=arn_string,
                write_bucket=f"{defaults.DEFAULT_BUCKET_WRITE}-{region_name}-{account_id}",
                read_bucket=f"{defaults.DEFAULT_BUCKET_READ}-{region_name}-{account_id}")

    # Check if config_file is an S3 URI
    if config_file.startswith("s3://"):
        try:
            # Parse S3 URI
            s3_client = boto3.client('s3')
            bucket, key = config_file.replace("s3://", "").split("/", 1)

            # Get object from S3 and load YAML
            response = s3_client.get_object(Bucket=bucket, Key=key)
            content = response["Body"].read().decode('utf-8')
            
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
            content = response.text
        except requests.exceptions.RequestException as e:
            print(f"Error loading config from HTTPS URL: {e}")
            raise
    else:
        # Assume local file system if not S3 or HTTPS
        try:
            content = Path(config_file).read_text()
        except Exception as e:
            print(f"Error loading config from local file system: {e}")
            raise
    
    content = content.format(**args)
    config = yaml.safe_load(content)
    return config

def count_tokens(text: str) -> int:
    global _tokenizer
    return _tokenizer.count_tokens(text)

def process_item(item, prompt_template_keys: List, prompt_fmt: str) -> Dict:
    args = {}
    for k in prompt_template_keys:
        v = _normalize(item[k])
        args[k] = v
        args[f"{k}_len"] = _tokenizer.count_tokens(v)
    prompt = prompt_fmt.format(**args)
    prompt_len = count_tokens(prompt)
    return args | {
        "prompt": prompt,
        "prompt_len": prompt_len
    }

def nt_to_posix(p: str) -> str:
    return p.replace("\\", "/")

# Function to write data to S3
def write_to_s3(data, bucket_name, dir1, dir2, file_name):

    # Initialize S3 client
    s3_client = boto3.client('s3')

    # Construct the S3 file path
    s3_file_path = posixpath.join(nt_to_posix(dir1), nt_to_posix(dir2), file_name)
    logger.debug(f"write_to_s3, s3_file_path={s3_file_path}")
    try:
        # Write the JSON data to the S3 bucket
        s3_client.put_object(Bucket=bucket_name, Key=s3_file_path, Body=data)
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
        logger.debug(f"read_from_s3, reading file from bucket={bucket_name}, key={s3_file_path}")
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
    logger.debug(f"get_s3_object, bucket_name={bucket}, key={key}")

    # Create an S3 client
    s3_client = boto3.client('s3')

    # Retrieve the object from S3
    response = s3_client.get_object(Bucket=bucket, Key=key)

    # Read the content of the file
    content = response['Body'].read().decode('utf-8')

    return content

# Function to list files in S3 bucket with a specific prefix
def list_s3_files(bucket, prefix, suffix='.json'):
    filter_key_by_suffix = lambda k,s: True if s is None else k.endswith(s)
    s3_client = boto3.client('s3')
    next_continuation_token = None
    
    return_list = []
    while True:
        if next_continuation_token is not None:
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=nt_to_posix(prefix), ContinuationToken=next_continuation_token)
        else:
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=nt_to_posix(prefix))
        return_list += [item['Key'] for item in response.get('Contents', []) if filter_key_by_suffix(item['Key'], suffix) is True]
        logger.info(f"found {len(return_list)} items in bucket={bucket}, prefix={prefix}, suffix={suffix}")
        if response['IsTruncated'] is True:            
            next_continuation_token = response['NextContinuationToken']
        else:
            break
    logger.info(f"there are total of {len(return_list)} items in bucket={bucket}, prefix={prefix}, suffix={suffix}")
    return return_list

def download_multiple_files_from_s3(bucket_name, prefix, local_dir):
    """Downloads files from an S3 bucket and a specified prefix to a local directory."""
    logger.info(f"download_multiple_files_from_s3, bucket_name={bucket_name}, prefix={prefix}, local_dir={local_dir}")
    s3_client = boto3.client('s3')

    # Ensure the local directory exists
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # List and download files
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        key_list =  list_s3_files(bucket_name, prefix, suffix=None)
        for file_key in key_list:
            logger.debug(f"file_key={file_key}, prefix={prefix}") 
            local_file_key = file_key.replace(prefix, "")
            parent_dir_in_s3 = os.path.dirname(local_file_key)
            logger.debug(f"local_file_key={local_file_key}, parent_dir_in_s3={parent_dir_in_s3}")
            # the first char for parent_dir_in_s3 would always be a '/' so skip that
            local_dir_to_create = os.path.join(local_dir, parent_dir_in_s3[1:])
            os.makedirs(local_dir_to_create, exist_ok = True)
            logger.debug(f"local_dir_to_create={local_dir_to_create}, local_file_key={local_file_key}")
            local_file_to_create = os.path.basename(local_file_key)
            if file_key.endswith('/'):
                logger.info(f"skipping file_key={file_key}")
                continue
            
            local_file_path = os.path.join(local_dir_to_create, local_file_to_create)
            logger.debug(f"bucket_name={bucket_name}, file_key={file_key}, local_file_path={local_file_path}")
            s3_client.download_file(bucket_name, file_key, local_file_path)
            logger.debug(f"download_multiple_files_from_s3, Downloaded: {local_file_path}")
    except Exception as e:
        logger.error(f"An error occurred while downloading from S3: {e}")
