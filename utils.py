import os
import re
import yaml
import boto3
import logging
import unicodedata
import globals as  g
from typing import Dict
from transformers import AutoTokenizer
from botocore.exceptions import NoCredentialsError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_tokenizer = None

# initializing tokenizer in case it is encoded as none type 
def _initialize_tokenizer():
    global _tokenizer
    try:
        if _tokenizer is None:
            ## use normal tokenizer -- change this variable LOCAL_TOKENIZER_DIR
            local_dir = g.TOKENIZER
            # Check if the tokenizer files exist locally
            if not os.path.exists(local_dir):
                # If not, download from S3
                bucket_name = g.READ_BUCKET_NAME ## DO IT from config 
                prefix = g.TOKENIZER_DIR_S3
                _download_from_s3(bucket_name, prefix, local_dir)
            # Load the tokenizer from the local directory
            _tokenizer = AutoTokenizer.from_pretrained(local_dir)
    except Exception as e:
        logger.error(f"An error occurred while initializing the tokenizer: {e}")


# utility functions
def load_config(config_file) -> Dict:
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)
    
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
    _initialize_tokenizer()
    return len(_tokenizer.encode(text))

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
        "question_len": len(_tokenizer.encode(question)),
        "context_len": len(_tokenizer.encode(context)),
    }

# Function to write data to S3
def write_to_s3(json_data, bucket_name, models_dir, model_name, file_name):

    # Initialize S3 client
    s3_client = boto3.client('s3')

    # Construct the S3 file path
    s3_file_path = os.path.join(models_dir, model_name, file_name)

    try:
        # Write the JSON data to the S3 bucket
        s3_client.put_object(Bucket=bucket_name, Key=s3_file_path, Body=json_data)
        return (f"s3://{bucket_name}/{s3_file_path}")
    except NoCredentialsError:
        logger.error("Error: AWS credentials not found.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

        
## function to read from s3
def read_from_s3(bucket_name, file_name):

    # Initialize S3 client
    s3_client = boto3.client('s3')

    # Construct the S3 file path
    s3_file_path = os.path.join(file_name)

    try:
        # Fetch the object from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_file_path)
        
        return response['Body'].read().decode('utf-8')
    except NoCredentialsError:
        logger.error("Error: AWS credentials not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

## gets a single s3 file
def get_s3_object(s3_path):
    # Regular expression to extract bucket name and key from the full S3 path
    match = re.match(r's3://([^/]+)/(.+)', s3_path)
    if not match:
        raise ValueError("Invalid S3 path")

    bucket_name, key = match.groups()

    # Create an S3 client
    s3_client = boto3.client('s3')

    # Retrieve the object from S3
    response = s3_client.get_object(Bucket=bucket_name, Key=key)

    # Read the content of the file
    content = response['Body'].read().decode('utf-8')

    return content
