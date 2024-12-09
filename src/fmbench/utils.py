import re
import os
import yaml
import math
import boto3
import shutil
import logging
import requests
import tempfile
import posixpath
import unicodedata
from pathlib import Path
import concurrent.futures
from fmbench import globals
from fmbench import defaults
from transformers import AutoTokenizer
from typing import Union, Dict, List, Tuple
from botocore.exceptions import NoCredentialsError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# utility functions
def load_config(config_file: Union[Path, str]) -> Dict:
    """
    Load configuration from a local file or an S3 URI.

    :param config_file: Path to the local file or S3 URI (s3://bucket/key)
    :return: Dictionary with the loaded configuration
    """
    session = boto3.session.Session()
    region_name = session.region_name
    if region_name is None:
        print(
            f"boto3.session.Session().region_name is {region_name}, "
            f"going to use an metadata api to determine region name"
        )
        resp = requests.put(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
        )
        token = resp.text
        region_name = requests.get(
            "http://169.254.169.254/latest/meta-data/placement/region",
            headers={"X-aws-ec2-metadata-token": token},
        ).text
        os.environ["AWS_DEFAULT_REGION"] = region_name
    print(f"region_name={region_name}")
    caller = boto3.client("sts").get_caller_identity()
    account_id = caller.get("Account")
    arn_string = caller.get("Arn")
    role_arn_from_env = os.environ.get("FMBENCH_ROLE_ARN")
    if role_arn_from_env:
        print(f"role_arn_from_env={role_arn_from_env}, using it to set arn_string")
        arn_string = role_arn_from_env
    else:
        print(
            f"role_arn_from_env={role_arn_from_env}, using current sts caller identity to set arn_string"
        )
        # if this is an assumed role then remove the assumed role related pieces
        # because we are also using this role for deploying the SageMaker endpoint
        # arn:aws:sts::015469603702:assumed-role/SSMDefaultRoleForOneClickPvreReporting/i-0c5bba16a8b3dac51
        # should be converted to arn:aws:iam::015469603702:role/SSMDefaultRoleForOneClickPvreReporting
        if ":assumed-role/" in arn_string:
            role_name = arn_string.split("/")[-2]
            arn_string = f"arn:aws:iam::{account_id}:role/{role_name}"
            print(
                f"the sts role is an assumed role, setting arn_string to {arn_string}"
            )

    # check if the file is still parameterized and if so replace the parameters with actual values
    # if the file is not parameterized then the following statements change nothing
    write_bucket = os.environ.get(
        "WRITE_BUCKET", f"{defaults.DEFAULT_BUCKET_WRITE}-{region_name}-{account_id}"
    )
    # check if the tmp dir is used as an argument if local mode is set to yes. If so, then use that as the temp file directory
    # else use the default `tempfile` option
    tmp_dir = os.environ.get("TMP_DIR", tempfile.gettempdir())
    args = dict(
        region=session.region_name,
        role_arn=arn_string,
        read_tmpdir=os.path.join(tmp_dir, defaults.DEFAULT_LOCAL_READ),
        write_tmpdir=os.path.join(tmp_dir, defaults.DEFAULT_LOCAL_WRITE),
        write_bucket=write_bucket,
        read_bucket=f"{defaults.DEFAULT_BUCKET_READ}-{region_name}-{account_id}",
    )

    # Check if config_file is an S3 URI
    if config_file.startswith("s3://"):
        try:
            # Parse S3 URI
            s3_client = boto3.client("s3")
            bucket, key = config_file.replace("s3://", "").split("/", 1)

            # Get object from S3 and load YAML
            response = s3_client.get_object(Bucket=bucket, Key=key)
            content = response["Body"].read().decode("utf-8")

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


def load_main_config(config_file) -> Dict:
    config = load_config(config_file)
    # iterate through each experiment and populate the parameters section in the inference spec
    for i in range(len(config["experiments"])):
        # for the experiment at index i, look up the parameter set
        # retrieve the parameter set from the inference_parameter section
        # assign the parameters from that parameter set to a new key called
        # parameters in that experiment
        parameters = config["inference_parameters"][
            config["experiments"][i]["inference_spec"]["parameter_set"]
        ]
        config["experiments"][i]["inference_spec"]["parameters"] = parameters
        if config["experiments"][i].get("bucket") is None:
            config["experiments"][i]["bucket"] = config["aws"]["bucket"]
    local_mode = os.environ.get("LOCAL_MODE")
    tmp_dir = os.environ.get("TMP_DIR", tempfile.gettempdir())
    if local_mode == "yes":
        print("utils.py, local_mode = yes")
        config["aws"]["s3_and_or_local_file_system"] = "local"
        config["s3_read_data"]["s3_or_local_file_system"] = "local"
        if config["s3_read_data"].get("local_file_system_path") is None:
            config["s3_read_data"]["local_file_system_path"] = os.path.join(
                tmp_dir, defaults.DEFAULT_LOCAL_READ
            )
        if config["aws"].get("local_file_system_path") is None:
            config["aws"]["local_file_system_path"] = os.path.join(
                tmp_dir, defaults.DEFAULT_LOCAL_WRITE
            )
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
    return args | {"prompt": prompt, "prompt_len": prompt_len}


def nt_to_posix(p: str) -> str:
    return p.replace("\\", "/")


def is_read_local() -> str:
    is_read_local = globals.config.get("s3_read_data").get("s3_or_local_file_system")
    logger.debug(f"is_read_local: {is_read_local}")
    return is_read_local is not None and is_read_local == "local"


def _is_write_local_only():
    is_write_local_only = globals.config.get("aws").get("s3_and_or_local_file_system")
    logger.debug(f"is_write_local_only: {is_write_local_only}")
    return is_write_local_only is not None and is_write_local_only == "local"


def _upload_file_to_local(local_path: str, s3_path: str) -> None:
    dest = _get_local_write_path(s3_path)
    shutil.copy(local_path, dest)


def upload_file_to_s3(bucket: str, local_path: str, s3_path: str) -> None:
    if _is_write_local_or_both():
        _upload_file_to_local(local_path, s3_path)
    if _is_write_local_only():
        return

    s3 = boto3.resource("s3")
    try:
        s3.Bucket(bucket).upload_file(local_path, s3_path)
    except Exception as e:
        logger.error(f"upload_file_to_s3, An error occurred: {e}")


def _write_to_local_read(data, dir1, dir2, file_name):
    file_dir, actual_file_name = os.path.split(file_name)
    # remove the hf prefix before sending the file to the local fmbench-read directory
    file_dir = file_dir.removeprefix(globals.HF_DATASET_PREFIX)
    logger.info(
        f"File directory: {file_dir}, file name to be written locally: {actual_file_name}"
    )
    dir_path = _get_local_read_path(dir1 + "/" + dir2 + "/" + file_dir + "/")
    logger.info(f"dir path: {dir_path}")
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    file_path = dir_path + actual_file_name
    if isinstance(data, str):
        Path(file_path).write_text(data)
    else:
        Path(file_path).write_bytes(data)


def _write_to_local(data, dir1, dir2, file_name):
    dir = _get_local_write_path(dir1 + "/" + dir2 + "/")
    Path(dir).mkdir(parents=True, exist_ok=True)
    file = dir + file_name
    if type(data) == str:
        Path(file).write_text(data)
    else:
        Path(file).write_bytes(data)


# Function to write data to S3
def write_to_s3(data, bucket_name, dir1, dir2, file_name):
    if _is_write_local_or_both():
        # If the file name starts with 'hf:', then it means that the hugging face dataset
        # is going to be loaded at runtime and is supposed to be sent to the /tmp/fmbench-read/source_data
        # folder. In this case, it is written to the local fmbench-read directory.
        if file_name.startswith(globals.HF_DATASET_PREFIX):
            _write_to_local_read(data, dir1, dir2, file_name)
        else:
            _write_to_local(data, dir1, dir2, file_name)
    if _is_write_local_only():
        return

    # Initialize S3 client
    s3_client = boto3.client("s3")

    # Construct the S3 file path
    s3_file_path = posixpath.join(nt_to_posix(dir1), nt_to_posix(dir2), file_name)
    logger.debug(f"write_to_s3, s3_file_path={s3_file_path}")
    try:
        # Write the JSON data to the S3 bucket
        s3_client.put_object(Bucket=bucket_name, Key=s3_file_path, Body=data)
        return f"s3://{bucket_name}/{s3_file_path}"
    except NoCredentialsError:
        logger.error("write_to_s3, Error: AWS credentials not found.")
    except Exception as e:
        logger.error(f"write_to_s3, An error occurred: {e}")


# Function to concurrently write multiple data to S3
# input_list: List[Tuple[data, bucket_name, dir1, dir2, file_name]
def write_multiple_to_s3(input_list: List[Tuple[None, str, str, str, str]]) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        tasks = []
        for input_tuple in input_list:
            tasks.append(executor.submit(lambda: write_to_s3(*input_tuple)))

    for completed_task in concurrent.futures.as_completed(tasks):
        completed_task.result()


def _read_from_local(s3_file_path: str) -> str:
    try:
        s3_file_path = nt_to_posix(_get_local_read_path(s3_file_path))
        logger.debug(f"get_local_object, key={s3_file_path}")
        return Path(s3_file_path).read_bytes().decode("utf-8")
    except FileNotFoundError as e:
        logger.error(f"read_from_local, An error occurred: {e}")
        return None


## function to read from s3
def read_from_s3(bucket_name, s3_file_path):
    if is_read_local():
        return _read_from_local(s3_file_path)

    # Initialize S3 client
    s3_client = boto3.client("s3")
    s3_file_path = nt_to_posix(s3_file_path)

    try:
        # Fetch the object from S3
        logger.debug(
            f"read_from_s3, reading file from bucket={bucket_name}, key={s3_file_path}"
        )
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_file_path)
        return response["Body"].read().decode("utf-8")
    except NoCredentialsError:
        logger.error("read_from_s3, Error: AWS credentials not found.")
        return None
    except Exception as e:
        logger.error(f"read_from_s3, An error occurred: {e}")
        return None


def _get_local_object(bucket: str, key: str, decode: bool) -> str:
    key = nt_to_posix(key)
    logger.debug(f"get_local_object, key={key}")
    if bucket == globals.config["s3_read_data"]["read_bucket"]:
        pathname = _get_local_read_path(key)
    else:
        pathname = _get_local_write_path(key)
    if decode:
        return Path(pathname).read_bytes().decode("utf-8")
    else:
        return Path(pathname).read_bytes()


## gets a single s3 file
def get_s3_object(bucket: str, key: str, decode="True") -> str:
    if is_read_local():
        return _get_local_object(bucket, key, decode)

    key = nt_to_posix(key)
    logger.debug(f"get_s3_object, bucket_name={bucket}, key={key}")

    # Create an S3 client
    s3_client = boto3.client("s3")

    # Retrieve the object from S3
    response = s3_client.get_object(Bucket=bucket, Key=key)

    # Read the content of the file
    if decode:
        content = response["Body"].read().decode("utf-8")
    else:
        content = response["Body"].read()

    return content


def _list_local_files(bucket, prefix, suffix):
    if bucket == globals.config["s3_read_data"]["read_bucket"]:
        dir = _get_local_read_path(prefix)
    else:
        dir = _get_local_write_path(prefix)
    logger.info(f"dir for listing local files: {dir}")
    # Recursively search for files with the suffix in all subdirectories
    path_list = list(Path(dir).glob("**/*" + suffix))
    pathname_list = [str(item) for item in path_list]
    if bucket == globals.config["s3_read_data"]["read_bucket"]:
        return_list = [
            item.replace(_get_local_read_path(), "") for item in pathname_list
        ]
    else:
        return_list = [
            item.replace(_get_local_write_path(), "") for item in pathname_list
        ]
    return return_list


# Function to list files in S3 bucket with a specific prefix
def list_s3_files(bucket, prefix, suffix=".json"):
    if is_read_local():
        return _list_local_files(bucket, prefix, suffix)

    filter_key_by_suffix = lambda k, s: True if s is None else k.endswith(s)
    s3_client = boto3.client("s3")
    next_continuation_token = None

    return_list = []
    while True:
        if next_continuation_token is not None:
            response = s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=nt_to_posix(prefix),
                ContinuationToken=next_continuation_token,
            )
        else:
            response = s3_client.list_objects_v2(
                Bucket=bucket, Prefix=nt_to_posix(prefix)
            )
        return_list += [
            item["Key"]
            for item in response.get("Contents", [])
            if filter_key_by_suffix(item["Key"], suffix) is True
        ]
        logger.info(
            f"found {len(return_list)} items in bucket={bucket}, prefix={prefix}, suffix={suffix}"
        )
        if response["IsTruncated"] is True:
            next_continuation_token = response["NextContinuationToken"]
        else:
            break
    logger.info(
        f"there are total of {len(return_list)} items in bucket={bucket}, prefix={prefix}, suffix={suffix}"
    )
    return return_list


def _normalize(text, form="NFC"):
    # The files in LongBench contain nonstandard or irregular Unicode.
    # For compatibility and safety we normalize them.
    return unicodedata.normalize(form, str(text))


def _is_write_local_or_both():
    is_write_local_or_both = globals.config.get("aws").get(
        "s3_and_or_local_file_system"
    )
    logger.debug(f"is_write_local_or_both: {is_write_local_or_both}")
    return is_write_local_or_both is not None and (
        is_write_local_or_both == "local" or is_write_local_or_both == "both"
    )


def _get_local_read_path(dir_or_file: str = None) -> str:
    if dir_or_file is not None:
        local_read_path = (
            globals.config["s3_read_data"]["local_file_system_path"] + "/" + dir_or_file
        )
    else:
        local_read_path = globals.config["s3_read_data"]["local_file_system_path"] + "/"
    logger.debug(f"local_read_path: {local_read_path}")
    return local_read_path


def _get_local_write_path(dir_or_file: str = None) -> str:
    if dir_or_file is not None:
        local_write_path = (
            globals.config["aws"]["local_file_system_path"] + "/" + dir_or_file
        )
    else:
        local_write_path = globals.config["aws"]["local_file_system_path"] + "/"
    logger.debug(f"local_write_path: {local_write_path}")
    return local_write_path


def _download_multiple_files_from_local_write_path(prefix, local_dir):
    src = _get_local_write_path(prefix)
    print(
        f"_download_multiple_files_from_local_write_path, prefix={prefix}, src={src}, local_dir={local_dir}"
    )
    shutil.copytree(src, local_dir, dirs_exist_ok=True)


def _download_multiple_files_from_local_read_path(prefix, local_dir):
    src = _get_local_read_path(prefix)
    print(
        f"_download_multiple_files_from_local_read_path, prefix={prefix}, src={src}, local_dir={local_dir}"
    )
    if Path(src).is_dir():
        shutil.copytree(src, local_dir, dirs_exist_ok=True)
    else:
        print(
            f"_download_multiple_files_from_local_read_path, {src} does not exist, no files to copy"
        )


def download_multiple_files_from_s3(bucket_name, prefix, local_dir):
    if _is_write_local_or_both():
        if bucket_name == globals.config["aws"]["bucket"]:
            return _download_multiple_files_from_local_write_path(prefix, local_dir)
        elif bucket_name == globals.config["s3_read_data"]["read_bucket"]:
            return _download_multiple_files_from_local_read_path(prefix, local_dir)
        else:
            logger.error(
                f"bucket_name={bucket_name} which does not match write bucket={globals.config['aws']['bucket']} "
                f"or read bucket={globals.config['s3_read_data']['read_bucket']}"
            )

    """Downloads files from an S3 bucket and a specified prefix to a local directory."""
    logger.info(
        f"download_multiple_files_from_s3, bucket_name={bucket_name}, prefix={prefix}, local_dir={local_dir}"
    )
    s3_client = boto3.client("s3")

    # Ensure the local directory exists
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # List and download files
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        key_list = list_s3_files(bucket_name, prefix, suffix=None)
        for file_key in key_list:
            logger.debug(f"file_key={file_key}, prefix={prefix}")
            local_file_key = file_key.replace(prefix, "")
            parent_dir_in_s3 = os.path.dirname(local_file_key)
            logger.debug(
                f"local_file_key={local_file_key}, parent_dir_in_s3={parent_dir_in_s3}"
            )
            # the first char for parent_dir_in_s3 would always be a '/' so skip that
            local_dir_to_create = os.path.join(local_dir, parent_dir_in_s3[1:])
            os.makedirs(local_dir_to_create, exist_ok=True)
            logger.debug(
                f"local_dir_to_create={local_dir_to_create}, local_file_key={local_file_key}"
            )
            local_file_to_create = os.path.basename(local_file_key)
            if file_key.endswith("/"):
                logger.info(f"skipping file_key={file_key}")
                continue

            local_file_path = os.path.join(local_dir_to_create, local_file_to_create)
            logger.debug(
                f"bucket_name={bucket_name}, file_key={file_key}, local_file_path={local_file_path}"
            )
            s3_client.download_file(bucket_name, file_key, local_file_path)
            logger.debug(
                f"download_multiple_files_from_s3, Downloaded: {local_file_path}"
            )
    except Exception as e:
        logger.error(f"An error occurred while downloading from S3: {e}")


class CustomTokenizer:
    """A custom tokenizer class"""

    TOKENS: int = 1000
    WORDS: int = 750
    HF_TOKEN_FNAME: str = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "scripts", "hf_token.txt"
    )
    if Path(HF_TOKEN_FNAME).is_file() is True:
        print(f"{HF_TOKEN_FNAME} file found, going to set HF_TOKEN env var")
        HF_TOKEN: str = Path(HF_TOKEN_FNAME).read_text().strip()
        os.environ["HF_TOKEN"] = HF_TOKEN
    else:
        print(f"{HF_TOKEN_FNAME} file not found")

    def __init__(self, bucket, prefix, local_dir, model_id, hf_model_id=None):
        print(
            f"CustomTokenizer, based on HF transformers, {bucket} "
            f"prefix: {prefix} local_dir: {local_dir}, model_id: {model_id}"
        )
        # Check if the tokenizer files exist in s3 and if not, use the autotokenizer
        download_multiple_files_from_s3(bucket, prefix, local_dir)
        # Load the tokenizer from the local directory
        all_files = [f for f in list(Path(local_dir).iterdir()) if f.name != ".keep"]
        dir_not_empty = len(all_files)
        print(f"CustomTokenizer, all_files = {all_files}")
        abs_path = Path(local_dir).absolute().resolve()
        if dir_not_empty > 0:
            print(
                f"loading the provided tokenizer from local_dir={local_dir}, abs_path={abs_path}"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
            print(
                f"successfully loaded the tokenizer using AutoTokenizer.from_pretrained from {local_dir}"
            )
        else:
            print(f"{local_dir} directory is empty")
            try:
                if hf_model_id:
                    print(f'going to download tokenizer from HF for "{hf_model_id}"')
                    self.tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
                    print(
                        f'successfully loaded the tokenizer using AutoTokenizer.from_pretrained from HF for "{hf_model_id}"'
                    )
                else:
                    print(f'going to download tokenizer from HF for "{model_id}"')
                    self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                    print(
                        f'successfully loaded the tokenizer using AutoTokenizer.from_pretrained from HF for "{model_id}"'
                    )
            except Exception as e:
                print(
                    f"exception while loading tokenizer from HuggingFace, exception={e}"
                )
                print(
                    f"no tokenizer provided, the {local_dir}, abs_path={abs_path} is empty, "
                    f"using default tokenizer i.e. {self.WORDS} words = {self.TOKENS} tokens"
                )
                self.tokenizer = None

    def count_tokens(self, text):
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text))
        else:
            return int(math.ceil((self.TOKENS / self.WORDS) * len(text.split())))


_tokenizer = CustomTokenizer(
    globals.READ_BUCKET_NAME,
    globals.TOKENIZER_DIR_S3,
    globals.TOKENIZER,
    globals.TOKENIZER_MODEL_ID,
    globals.HF_TOKENIZER_MODEL_ID,
)
