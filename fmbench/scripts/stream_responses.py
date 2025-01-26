import io
import time
import json
import boto3
import litellm
import logging
import botocore
from typing import Dict, Optional, List, Union

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the SageMaker runtime to invoke the endpoint
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
                logger.warning('Unknown event type:' + str(chunk))
                continue
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(chunk['PayloadPart']['Bytes'])


def get_response_stream(response_stream: Union[litellm.utils.CustomStreamWrapper, botocore.eventstream.EventStream],
                        start_time: float,
                        start_token: str,
                        stop_token: str,
                        is_sagemaker: bool = False) -> Dict:
    """
    Helper function to get the response streams from bedrock or sagemaker invocations
    and parse each as appropriate to calculate the Time To First Token (TTFT), Time Per Output Token (TPOT), 
    and Time To Last Token (TTLT)

    return: This function returns a dictionary containing the entire response, and the TTFT, TTLT, TPOT metrics
    """
    logger.info(f"get_response_stream, type(response_stream)={type(response_stream)}")
    sm_start_token = b"{"
    first_token_time: Optional[float] = None
    token_times: List[float] = []
    last_token_time = start_time
    response_text: str = ""
    TTFT: Optional[float] = None
    TPOT: Optional[float] = None
    TTLT: Optional[float] = None
    result: Optional[Dict] = None

    try:
        # get the event from the sagemaker or bedrock response streams
        event_iterator = LineIterator(response_stream) if is_sagemaker else response_stream

        for event in event_iterator:
            # if the response stream is from a sagemaker call, then use the 
            # line iterator to get the first token from the streaming response
            token_id: Optional[int] = None
            if is_sagemaker:
                if event != b'' and sm_start_token in event:
                    data = json.loads(event[event.find(sm_start_token):].decode('utf-8'))
                    #logger.info(f"data={data}")
                    token_id = data['token']['id']
                    if token_id != [-1]:
                        token_text = data['token']['text']
                    else:
                        token_text = None
                else:
                    continue
            else:
                # if the response stream is from a bedrock call, then get the chunks from the response
                # and the first token
                if hasattr(event, 'choices') and hasattr(event.choices[0], 'delta'):
                    token_text = event.choices[0].delta.get('content', '')
                else:
                    continue
            # record the current time
            current_time = time.perf_counter()
            if token_text and token_text != stop_token:
                if first_token_time is None:
                    first_token_time = current_time
                    # get the time to first token latency
                    TTFT = first_token_time - start_time
                    logger.info(f"Time to First Token: {TTFT:.6f} seconds")
                else:
                    # if the token is not the first token, then get the inter-token
                    # latency
                    token_time = current_time - last_token_time
                    # append all token times to get the time per output token
                    token_times.append(token_time)
                last_token_time = current_time
                response_text += token_text

            if stop_token and stop_token in response_text:
                logger.info(f"got the last token: {stop_token}")
                break
            elif token_id == [-1]:
                logger.info(f"end of stream because token id is {token_id}")
                break

        # Calculate TTLT at the reception of the last token
        current_time = time.perf_counter()
        TTLT = current_time - start_time
        logger.info(f"Time to Last Token (TTLT): {TTLT:.6f} seconds, total tokens received {len(token_times)}")

        if token_times:
            # get the average of all token times to compute the time per output token
            TPOT = sum(token_times) / len(token_times)
            logger.info(f"Time Per Output Token (TPOT): {TPOT:.6f} seconds")

        response_data = [{"generated_text": response_text}]
        response_json_str = json.dumps(response_data)
        result = {
            "TTFT": TTFT,
            "TPOT": TPOT,
            "TTLT": TTLT,
            "response": response_json_str
        }
    except Exception as e:
        logger.error(f"Error occurred while generating and computing metrics "
                     f"associated with the streaming response: {e}", exc_info=True)
        result = None

    logger.info(f"Final result: {result}")
    return result

