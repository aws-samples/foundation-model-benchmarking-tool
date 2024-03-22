import time
import json
import boto3
import logging
import anthropic
import requests as req
import botocore.session
from typing import Dict
import anthropic_bedrock
from itertools import groupby
from operator import itemgetter
from botocore.config import Config
from botocore.auth import SigV4Auth
from typing import Dict, List, Tuple
from fmbench.utils import count_tokens
from botocore.awsrequest import AWSRequest
from anthropic_bedrock import AnthropicBedrock
from fmbench.scripts.fmbench_predictor import FMBenchPredictor, FMBenchPredictionResponse

## set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


## Represents the class to predict using a bedrock rest API 
class BedrockPredictor(FMBenchPredictor):

    # overriding abstract method
    def __init__(self, endpoint_name: str):

        ## initiliazing the bedrock client
        bedrock_client = boto3.client('bedrock-runtime')

        self._predictor: Optional[bedrock_client] = None
        self._endpoint_name: str = endpoint_name

        try:
            # Create a bedrock runtime client 
            self._predictor = bedrock_client

        except Exception as e:
            logger.error(f"create_predictor, exception occured while creating predictor for endpoint_name={self._endpoint_name}, exception={e}")
        logger.info(f"__init__ self._predictor={self._predictor}")
        
    def get_prediction(self, payload: Dict) -> FMBenchPredictionResponse:
        response_json = None
        latency = None
        response = None
        inference_params = None
        prompt_tokens = None
        completion_tokens = None

        prompt_input_data = payload['inputs']

        ## extract the input data prompts here using our normal count tokenizer to count the number of input tokens
        prompt_tokens = count_tokens(payload['inputs']) ## NEED TO ADD BEDROCK SUPPORT FOR EVERY MODEL HERE TOO - TODO

        # Get the AWS region dynamically
        aws_region = boto3.Session().region_name

        ## initialize an anthropic client to get token counts for claude models
        client = AnthropicBedrock()

        # Build the REST API URL based on the region and self.endpoint_name
        bedrock_rest_api_url = f"https://bedrock-runtime.{aws_region}.amazonaws.com/model/{self.endpoint_name}/invoke"
        logger.info(f"REST API URL created based on bedrock model id '{self.endpoint_name}' : {bedrock_rest_api_url}")

        ## ------------------------------------------------------ Default parameters and prompt formats ------------------------------------------------------
        ## inference parameters for each model type - need to abstract this out
        inference_params = {
            "claude_v2": {
                "max_tokens_to_sample": 300,
                "top_p": 1,
                "top_k": 250,
                "stop_sequences": ["\n\nHuman:"],
            },
            "claude_3": {
                "max_tokens": 2000,
            },
            "llama_2": {
                "max_gen_len": 512,
                "temperature": 0.5,
                "top_p": 0.9,
            },
            "mixtral": {
                "max_tokens": 200,
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": 50,
            },
            "cohere": {
                "temperature": 0.9,
                "p": 1,
                "k": 0,
            },
            "ai21": {
                "maxTokens": 400,
                "temperature": 0.9,
                "topP": 0.9,
                "stopSequences": [],
                "countPenalty": {"scale": 0},
                "presencePenalty": {"scale": 0},
                "frequencyPenalty": {"scale": 0},
            },
        }

        # Define formats for different models
        claude_v2_format = {
            "prompt": f"\n\nHuman:{prompt_input_data}\n\nAssistant:",
            "anthropic_version": "bedrock-2023-05-31",
            **inference_params["claude_v2"],
        }

        claude_3_format = {
            "messages": [{"role": "user", "content": prompt_input_data}],
            "anthropic_version": "bedrock-2023-05-31",
            **inference_params["claude_3"],
        }

        llama_2_format = {
            "prompt": prompt_input_data,
            **inference_params["llama_2"],
        }

        mixtral_format = {
            "prompt": f"<s>[INST] {prompt_input_data} [/INST]",
            **inference_params["mixtral"],
        }

        cohere_format = {
            "prompt": prompt_input_data,
            **inference_params["cohere"],
        }

        ai21_format = {
            "prompt": prompt_input_data,
            **inference_params["ai21"],
        }

        ## ------------------------------------------------------ model mapping to payload formats ------------------------------------------------------

        # Define a dictionary with payload formats for different models associated to the model id
        payload_formats = {
            "anthropic.claude-3-haiku-20240307-v1:0": claude_3_format,
            "anthropic.claude-3-sonnet-20240229-v1:0": claude_3_format,
            "anthropic.claude-v2": claude_v2_format,
            "anthropic.claude-instant-v1": claude_v2_format,
            "anthropic.claude-v2:1": claude_v2_format,
            "meta.llama2-13b-chat-v1": llama_2_format,
            "meta.llama2-70b-chat-v1": llama_2_format,
            "mistral.mistral-7b-instruct-v0:2": mixtral_format,
            "mistral.mixtral-8x7b-instruct-v0:1": mixtral_format,
            "cohere.command-text-v14": cohere_format,
            "cohere.command-light-text-v14": cohere_format,
            "amazon.titan-embed-text-v1": {"inputText": prompt_input_data},
            "amazon.titan-text-lite-v1": {"inputText": prompt_input_data},
            "amazon.titan-text-express-v1": {"inputText": prompt_input_data},
            "ai21.j2-ultra-v1": ai21_format,
            "ai21.j2-mid-v1": ai21_format,
        }

         # Get the payload format based on self.endpoint_name
        text_payload = payload_formats.get(self.endpoint_name)
        if not text_payload:
            logger.error(f"Unrecognized model endpoint: {self.endpoint_name}. Use a valid bedrock model id. Refer to supported models here: https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html")
        ## -------------------------------------------------------------------------------------------------------------------------------------

        # Converting the payload dictionary into a JSON-formatted string to be sent in the HTTP request
        request_payload = text_payload
        request_body = json.dumps(request_payload)

        # Creating an AWSRequest object for a POST request with the service specified endpoint, JSON request body, and HTTP headers
        request = AWSRequest(method='POST',
                            url=bedrock_rest_api_url,
                            data=request_body,
                            headers={'content-type': 'application/json'})

        # Initializing a botocore session
        session = botocore.session.Session()

        # Adding a SigV4 authentication information to the AWSRequest object, signing the request
        sigv4 = SigV4Auth(session.get_credentials(), 'bedrock', 'us-east-1')
        sigv4.add_auth(request)

        # Prepare the request by formatting it correctly
        prepped = request.prepare()

        # start the latency metric timer here
        st = time.perf_counter()
        # Send the HTTP POST request to the prepared URL with the specified headers & JSON-formatted request body, storing the response
        response = req.post(prepped.url, headers=prepped.headers, data=request_body)
        # recording the latency at the end of bedrock prediction
        latency = time.perf_counter() - st

        if response.status_code == 200:
            response_body = response.content.decode('utf-8')
            response_json = json.loads(response_body)

            if isinstance(response_json, list):
                response_json = response_json[0]
                # response_json = response_json

            # add a key called completion, if not there, this is for haiku, sonnet
            if response_json is not None:
                if response_json.get("generated_text") is None:
                    if "content" in response_json:
                        content_items = response_json["content"]
                        response_text = ""
                        for content_item in content_items:
                            if content_item["type"] == "text":
                                response_text += content_item["text"]
                        response_json["generated_text"] = response_text
                    ## this is for ai21
                    elif "generatedToken" in response_json:
                        # If the response contains generatedToken, create the completion string
                        completion = ""
                        for token_dict in response_json["generatedToken"]:
                            completion += token_dict["generatedToken"]["token"]
                        response_json["generated_text"] = completion
                    ## this is for mistral, llama
                    elif "outputs" in response_json: 
                        # If the response contains outputs, get the text from the first output
                        if response_json["outputs"] and isinstance(response_json["outputs"], list):
                            first_output = response_json["outputs"][0]
                            if "text" in first_output:
                                response_json["generated_text"] = first_output["text"]
                    ## this is for claude v2, v2:1 and instant models
                    elif "completion" in response_json:
                        # Extract generated text from "completion" key
                        response_json["generated_text"] = response_json["completion"]
                    else:
                        logger.warning(f"get_prediction, response_json does not contain 'content', 'generatedToken', or 'outputs' key: {response_json}")
                
                
                
                
                ## this is the PAYLOAD PROMPT COUNT BASED ON ALL MODELS ON BEDROCK 
                ## extract the inputs from the payload for the bedrock specific prompt plug in below in payload formats
                ## this is the case for llama on bedrock
                if "prompt_token_count" in response_json:
                    prompt_tokens = response_json["prompt_token_count"]
                    logger.info(f"LLAMA BEDROCK PROMPT TOKENS: {prompt_tokens}")
                ## this is the case for titan on bedrock
                elif "inputTextTokenCount" in response_json:
                    completion_tokens = response_json["inputTextTokenCount"]
                    logger.info(f"TITAN PROMPT TOKENS: {prompt_tokens}")
                ## this is the case for all claude models
                elif "claude" in bedrock_rest_api_url:
                    completion_tokens = client.count_tokens(prompt_input_data) 
                    logger.info(f"CLAUDE PROMPT TOKENS: {prompt_tokens}")
                else:
                    logger.info(f"Counting tokens using the default hf tokenizer......")
                    prompt_tokens = count_tokens(prompt_input_data)


                ## this is the TOKEN COMPLETION COUNT FOR ALL MODELS ON BEDROCK -- llama and titan. CLAUDE ADDED AS AN ANTHROPIC TOKEN COUNTER CLIENT
                if "generation_token_count" in response_json:
                    completion_tokens = response_json["generation_token_count"]
                    logger.info(f"Llama completion tokens: {completion_tokens}")
                elif "tokenCount" in response_json:
                    completion_tokens = response_json["tokenCount"]
                    logger.info(f"Titan completion tokes: {completion_tokens}")
                elif "claude" in bedrock_rest_api_url:
                    completion_tokens = client.count_tokens(response_json['generated_text'])
                    logger.info(f"Claude completion tokens: {completion_tokens}")                               
                else: ## for all other models
                    completion = response_json.get("generated_text", "")
                    logger.info(f"Counting tokens using the default hf tokenizer......")
                    completion_tokens = count_tokens(completion)
        else:
            logger.error(f"get_prediction, received non-200 status code {response.status_code} from predictor={self._endpoint_name}")

        return FMBenchPredictionResponse(response_json=response_json, latency=latency, completion_tokens=completion_tokens, prompt_tokens=prompt_tokens)
    
    def calculate_cost(self, duration: float, metrics: Dict) -> float: ## ---> TO IMPLEMENT
        """
        Represents the function to calculate the cost of each experiment run for Bedrock.
        
        Args:
            duration (float): The duration of the experiment run in seconds.
            metrics (Dict): A dictionary containing metrics related to the experiment run.
        
        Returns:
            float: The cost of the experiment run.
        """
        # Implement the logic to calculate the cost based on the duration and metrics
        # This could involve factors like the number of tokens generated, the model size, etc.
        # For simplicity, let's assume a fixed cost per second
        pass 

    @property
    def endpoint_name(self) -> str:
        """The endpoint name property."""
        return self._endpoint_name
    
    def calculate_cost(self, instance_type, config: dict, duration: float, metrics: dict) -> float:
        """Represents the function to calculate the cost for Bedrock experiments."""
        experiment_cost = 0.0

        if metrics:
            prompt_tokens = metrics.get("all_prompts_token_count", 0)
            completion_tokens = metrics.get("all_completions_token_count", 0)

            # Retrieve the pricing information for the instance type
            instance_pricing = config['pricing'].get(instance_type, {})
            input_token_price_per_thousand = 0
            output_token_price_per_thousand = 0

            for pricing_dict in instance_pricing:
                logger.info(f"pricing dict: {pricing_dict}")
                if 'input-per-1k-tokens' in pricing_dict:
                    input_token_price_per_thousand = pricing_dict['input-per-1k-tokens']
                    logger.info(f"input per 1k token pricing: {input_token_price_per_thousand}")
                if 'output-per-1k-tokens' in pricing_dict:
                    output_token_price_per_thousand = pricing_dict['output-per-1k-tokens']
                    logger.info(f"output per 1k token pricing: {output_token_price_per_thousand}")

            # Calculate cost based on the number of input and output tokens
            input_token_cost = (prompt_tokens / 1000.0) * input_token_price_per_thousand
            output_token_cost = (completion_tokens / 1000.0) * output_token_price_per_thousand

            experiment_cost = input_token_cost + output_token_cost

        return experiment_cost
    
def create_predictor(endpoint_name: str):
    return BedrockPredictor(endpoint_name)