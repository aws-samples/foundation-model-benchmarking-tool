import json
import os

import torch
from transformers_neuronx import NeuronAutoModelForCausalLM
from transformers_neuronx.config import NeuronConfig, GenerationConfig, QuantizationConfig, ContinuousBatchingConfig
from transformers import AutoTokenizer
import numpy as np
import time

import triton_python_backend_utils as pb_utils

_MODEL_ARGS_FILENAME = "model.json"
_MAX_MODEL_LEN = 8192

class TritonPythonModel:

  def initialize(self, args):
    self.logger = pb_utils.Logger
    self.model_config = json.loads(args["model_config"])
    text_output_config = pb_utils.get_output_config_by_name(self.model_config, "text_output")
    self.text_output_dtype = pb_utils.triton_string_to_numpy(text_output_config["data_type"])
    self.load_model()
      

  @staticmethod
  def auto_complete_config(auto_complete_model_config):
      
    inputs = [
    {"name": "text_input", "data_type": "TYPE_STRING", "dims": [1]},
    {
        "name": "sampling_parameters",
        "data_type": "TYPE_STRING",
        "dims": [1],
        "optional": True,
    }
    ]
    outputs = [{"name": "text_output", "data_type": "TYPE_STRING", "dims": [-1]}]

    config = auto_complete_model_config.as_dict()
    input_names = []
    output_names = []
    for input in config['input']:
      input_names.append(input['name'])
    for output in config['output']:
      output_names.append(output['name'])

    for input in inputs:
      if input['name'] not in input_names:
          auto_complete_model_config.add_input(input)
    for output in outputs:
      if output['name'] not in output_names:
          auto_complete_model_config.add_output(output)

    auto_complete_model_config.set_model_transaction_policy(dict(decoupled=False))
    auto_complete_model_config.set_max_batch_size(1)
    auto_complete_model_config.set_dynamic_batching()

    return auto_complete_model_config
  
  def load_model(self):
    self.logger.log_info("Enter: load_model")

    max_batch_size = int(self.model_config.get('max_batch_size', 1))
    assert (
        max_batch_size >= 1 
    ), "max_batch_size must be >= 1 for dynamic batching"

    self.using_decoupled = pb_utils.using_decoupled_model_transaction_policy(self.model_config) 
    assert (
        not self.using_decoupled 
    ), "Python backend must be configured to not use decoupled model transaction policy"

    model_args_filepath = os.path.join( 
        pb_utils.get_model_dir(), _MODEL_ARGS_FILENAME
    )
    assert os.path.isfile(
        model_args_filepath
    ), f"'{_MODEL_ARGS_FILENAME}' containing model args must be provided in '{pb_utils.get_model_dir()}'"
    with open(model_args_filepath) as file:
        model_args = json.load(file)

    model_location = model_args.pop("model")
  
    neuron_config_dict = model_args.pop("neuron_config", {})
    continuous_batching = neuron_config_dict.pop("continuous_batching", {})
    if "batch_size_for_shared_caches" in continuous_batching:
      model_args["batch_size"] = int(continuous_batching.get("batch_size_for_shared_caches"))
      neuron_config_dict["continuous_batching"] = ContinuousBatchingConfig(**continuous_batching)
    
    quant = neuron_config_dict.pop("quant", {})
    if "quant_dtype" in quant and "dequant_dtype" in quant:
      neuron_config_dict["quant"] = QuantizationConfig(**quant)
    
    on_device_embedding = neuron_config_dict.pop("on_device_embedding", {})
    if on_device_embedding:
      neuron_config_dict["on_device_embedding"] = GenerationConfig(**on_device_embedding)

    if neuron_config_dict:
      model_args["neuron_config"] = NeuronConfig(**neuron_config_dict)

    self.batch_size = model_args.get("batch_size", 1)

    tokenizer_location = model_args.pop("tokenizer", model_location)
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_location)

    self.logger.log_info(f"normalized model_args: {model_args}")
    self.model = NeuronAutoModelForCausalLM.from_pretrained(model_location, **model_args)

    self.logger.log_info(f"Move model to Neuron device")
    self.model.to_neuron()
    self.logger.log_info("Model moved to Neuron device")


  def get_sampling_params_dict(self, params_json):              
    params_dict = json.loads(params_json) if params_json else {}

    float_keys = [
        "temperature",
        "top_p"
    ]
    for k in float_keys:
        if k in params_dict:
            params_dict[k] = float(params_dict[k])
        
    int_keys = ["sequence_length", "top_k"]
    for k in int_keys:
        if k in params_dict:
            params_dict[k] = int(params_dict[k])

    if not params_dict:
        params_dict["sequence_length"] = _MAX_MODEL_LEN
        params_dict["top_k"] = 50
    elif "sequence_length" not in params_dict:
        params_dict["sequence_length"] = _MAX_MODEL_LEN

    return params_dict

  def execute(self, requests):
    output_dtype = self.text_output_dtype

    responses = []
    requests_start = time.time()
    r_prompts = []
    r_params = []
    for request in requests:
      prompts = pb_utils.get_input_tensor_by_name(request, "text_input").as_numpy().flatten()
      prompts = prompts.tolist()
      prompts = [ p.decode("utf-8") if isinstance(p, bytes) else p for p in prompts]

      r_prompts.append(prompts)

      parameters_input_tensor = pb_utils.get_input_tensor_by_name(request, "sampling_parameters")
      if parameters_input_tensor:
        parameters = parameters_input_tensor.as_numpy().flatten()
        parameters = parameters.tolist()[0] # assume uniform sampling parameters in batch
        parameters = parameters.decode('utf-8') if isinstance(parameters, bytes) else parameters
      else:
        parameters = request.parameters()
      params = self.get_sampling_params_dict(parameters)
      r_params.append(params)

    self.logger.log_info(f"Processing {len(r_params)} requests")
    r_cuts = []
    for r_index in range(len(r_params) - 1):
      if r_params[r_index] != r_params[r_index + 1]:
        r_cuts.append(r_index+1)
            
    r_cuts.append(len(r_params))
    self.logger.log_info(f"Request cuts: {r_cuts}")

    r_index = 0
    generated_text_seqs_list = []
    for r_cut in r_cuts:
      all_prompts = [ prompt for prompts_list in r_prompts[r_index:r_cut] for prompt in prompts_list ]
      self.logger.log_info(f"Request cut total prompts: {len(all_prompts)}")
      params = r_params[r_index]

      self.logger.log_info(f"Request cut params: {params}")

      with torch.inference_mode():
        while all_prompts:
          batch_prompts = all_prompts[:self.batch_size]
          batch_prompts_size = len(batch_prompts)
          batch_prompts.extend(batch_prompts[-1] for _ in range(self.batch_size - batch_prompts_size))

          start_time = time.time()
          input_ids = self.tokenizer.batch_encode_plus(batch_prompts, return_tensors="pt")['input_ids']
          generated_token_seqs = self.model.sample(input_ids, **params)
          generated_token_seqs = generated_token_seqs[:batch_prompts_size]
          generated_text_seqs = self.tokenizer.batch_decode(generated_token_seqs, skip_special_tokens=True)
          assert isinstance(generated_text_seqs, list)
          assert len(generated_text_seqs) == batch_prompts_size
          generated_text_seqs_list.extend(generated_text_seqs)
          end_time = time.time()
          self.logger.log_info(f"model execution time: {end_time - start_time} for batch size: {batch_prompts_size}")
          all_prompts = all_prompts[batch_prompts_size:]

        r_index = r_cut

    text_output_seqs_list = []
    i = 0
    for p_list in r_prompts:
      p_list_len = len(p_list)
      text_output_seqs = generated_text_seqs_list[i:i+p_list_len]
      text_output_seqs_list.append(text_output_seqs)
      i += p_list_len

    for text_output_seqs in text_output_seqs_list:
      out_tensor = pb_utils.Tensor("text_output", np.array(text_output_seqs).astype(output_dtype))
      inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])

      responses.append(inference_response)
    requests_end = time.time()
    self.logger.log_info(f"requests execution time: {requests_end - requests_start}")

    return responses

  def finalize(self):
    self.logger.log_info("Cleaning up...")
    