import io
import time
import json
import boto3
import logging
from typing import Dict, Optional, List

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


def get_sagemaker_response_stream_token_metrics(response_stream, stop_token: str) -> Dict:
    event_stream = response_stream['Body']
    start_json: str = b'{'
    start_time: float = time.perf_counter()
    first_token_time: Optional[float] = None
    token_times: Optional[List[float]] = []
    last_token_time = start_time
    response_text: str = ""
    TTFT: Optional[float] = None
    TPOT: Optional[float] = None
    TTLT: Optional[float] = None
    result: Optional[Dict] = None

    try:
        for line in LineIterator(event_stream):
            if line != b'' and start_json in line:
                data = json.loads(line[line.find(start_json):].decode('utf-8'))
                if data['token']['text'] != stop_token:
                    current_time = time.perf_counter()

                    if first_token_time is None:
                        first_token_time = current_time
                        TTFT = first_token_time - start_time
                        logger.info(f"Time to First Token: {TTFT:.3f} seconds")
                    else:
                        token_time = current_time - last_token_time
                        token_times.append(token_time)

                    last_token_time = current_time
                    response_text += data['token']['text']
                else:
                    # Calculate TTLT at the reception of the last token
                    current_time = time.perf_counter()
                    TTLT = current_time - start_time
                    logger.info(f"Time to Last Token (TTLT): {TTLT:.3f} seconds")
                    break

        if token_times:
            TPOT = sum(token_times) / len(token_times)
            logger.info(f"Time Per Output Token (TPOT): {TPOT:.3f} seconds")

        response_data = [{"generated_text": response_text}]
        response_json_str = json.dumps(response_data)
        result = {
            "TTFT": TTFT,
            "TPOT": TPOT,
            "TTLT": TTLT,
            "Response": response_json_str
        }
    except Exception as e:
        logger.error(f"Error occurred while generating and computing metrics associated with the streaming response: {e}")
        result = None

    # Additional logging for diagnosis
    logger.info(f"Final result: {result}")
    return result

def get_bedrock_response_stream_token_metrics(response_stream, stop_token: str) -> Dict:
    start_time: float = time.perf_counter()
    first_token_time: Optional[float] = None
    token_times: Optional[List[float]] = []
    last_token_time = start_time
    response_text: str = ""
    TTFT: Optional[float] = None
    TPOT: Optional[float] = None
    TTLT: Optional[float] = None
    result: Optional[Dict] = None

    try:
        for chunk in response_stream:
            logger.info(f"chunk found")
            if hasattr(chunk, 'choices') and hasattr(chunk.choices[0], 'delta'):
                token_text = chunk.choices[0].delta.get('content', '')
                current_time = time.perf_counter()
                if token_text:
                    if first_token_time is None:
                        logger.info(f"computing the first token time")
                        first_token_time = current_time
                        TTFT = first_token_time - start_time
                        logger.info(f"Time to First Token: {TTFT:.3f} seconds")
                    else:
                        token_time = current_time - last_token_time
                        token_times.append(token_time)

                    last_token_time = current_time
                    response_text += token_text

                    # Check for stop token
                    if stop_token and stop_token in response_text:
                        logger.info(f"got the last token: {stop_token}")
                        break

        # Calculate TTLT at the reception of the last token
        current_time = time.perf_counter()
        TTLT = current_time - start_time
        logger.info(f"Time to Last Token (TTLT): {TTLT:.3f} seconds")

        if token_times:
            TPOT = sum(token_times) / len(token_times)
            logger.info(f"Time Per Output Token (TPOT): {TPOT:.3f} seconds")

        response_data = [{"generated_text": response_text}]
        response_json_str = json.dumps(response_data)
        result = {
            "TTFT": TTFT,
            "TPOT": TPOT,
            "TTLT": TTLT,
            "Response": response_json_str
        }
    except Exception as e:
        logger.error(f"Error occurred while generating and computing bedrock metrics associated with the streaming response: {e}", exc_info=True)
        result = None

    # Additional logging for diagnosis
    logger.info(f"Final result: {result}")
    return result
