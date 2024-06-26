import io
import time
import json
import boto3
import logging
from typing import Dict, Optional

# set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the sagemaker runtime to invoke the endpoint 
# with a response stream using line iterator
sagemaker_runtime = boto3.client('sagemaker-runtime')


class LineIterator:
    """
    A helper class for parsing the byte stream input.

    The output of the model will be in the following format:
    ```
    b'{"outputs": [" a"]}\n'
    b'{"outputs": [" challenging"]}\n'
    b'{"outputs": [" problem"]}\n'
    ...
    ```

    While usually each PayloadPart event from the event stream will contain a byte array 
    with a full json, this is not guaranteed and some of the json objects may be split across
    PayloadPart events. For example:
    ```
    {'PayloadPart': {'Bytes': b'{"outputs": '}}
    {'PayloadPart': {'Bytes': b'[" problem"]}\n'}}
    ```

    This class accounts for this by concatenating bytes written via the 'write' function
    and then exposing a method which will return lines (ending with a '\n' character) within
    the buffer via the 'scan_lines' function. It maintains the position of the last read 
    position to ensure that previous bytes are not exposed again. 
    """

    def __init__(self, stream):
        self.byte_iterator = iter(stream)
        self.buffer = io.BytesIO()
        self.read_pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            self.buffer.seek(self.read_pos)
            line = self.buffer.readline()
            if line and line[-1] == ord('\n'):
                self.read_pos += len(line)
                return line[:-1]
            try:
                chunk = next(self.byte_iterator)
            except StopIteration:
                if self.read_pos < self.buffer.getbuffer().nbytes:
                    continue
                raise
            if 'PayloadPart' not in chunk:
                print('Unknown event type:' + chunk)
                continue
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(chunk['PayloadPart']['Bytes'])


def get_sagemaker_realtime_response_stream(sagemaker_runtime, endpoint_name, payload):
    response_stream = sagemaker_runtime.invoke_endpoint_with_response_stream(
        EndpointName=endpoint_name,
        Body=json.dumps(payload), =
        ContentType="application/json",
        CustomAttributes='accept_eula=true'
    )
    return response_stream


def get_sagemaker_response_stream(response_stream) -> Dict:
    event_stream = response_stream['Body']
    start_json = b'{'
    stop_token = '</s>'
    start_time = time.time()
    first_token_time = None
    token_times = []
    last_token_time = start_time
    response_text = ""
    result: Optional[Dict] = None
    ttft = None
    tpot = None

    for line in LineIterator(event_stream):
        if line != b'' and start_json in line:
            data = json.loads(line[line.find(start_json):].decode('utf-8'))
            if data['token']['text'] != stop_token:
                current_time = time.time()

                if first_token_time is None:
                    first_token_time = current_time
                    ttft = first_token_time - start_time
                    logger.info(f"Time to First Token: {ttft:.3f} seconds")
                else:
                    token_time = current_time - last_token_time
                    token_times.append(token_time)

                last_token_time = current_time
                response_text += data['token']['text']

    if token_times:
        tpot = sum(token_times) / len(token_times)
        logger.info(f"Time Per Output Token (TPOT): {tpot:.3f} seconds")

    response_data = [{"generated_text": response_text}]
    response_json_str = json.dumps(response_data)
    result = {
        "TTFT": ttft,
        "TPOT": tpot,
        "Response": response_json_str
    }
    return result
