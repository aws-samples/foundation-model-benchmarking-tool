import os
import torch
import logging
import argparse
from transformers.models.opt import OPTForCausalLM
from transformers_neuronx.module import save_pretrained_split
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_REPO: str = "meta-llama"
MODEL_ID: str = "Meta-Llama-3-1-70bB-Instruct"
NEURON_VER: str = "2.18.2"

root = logging.getLogger()
if root.handlers:
    for handler in root.handlers:
        root.removeHandler(handler)
logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    
    if 'HF_TOKEN' not in os.environ:
        logger.info('Hugging Face Hub token is missing')
        exit(-1)

    # Define and parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", "-m", 
        type=str, 
        default=f"{MODEL_REPO}/{MODEL_ID}",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--save-path", "-s",
        type=str,
        default=f"../{NEURON_VER}/model_store/{MODEL_ID}/{MODEL_ID}-split/",
        help="Output directory for downloaded model files",
    )
    args = parser.parse_args()
    logger.info(f"args={args}")

    save_path = os.makedirs(args.save_path, exist_ok=True)
    logger.info(f"Save path defined for the model: {save_path}")

    # Load HuggingFace model
    logger.info(f"Going to load the hugging face model {args.model_name}")
    hf_model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                                    low_cpu_mem_usage=True)
    logger.info(f"Successfully loaded the model {args.model_name}")


    # Save the model
    logger.info('Going to split and save the model')
    save_pretrained_split(hf_model, args.save_path)
    logger.info('Model splitted and saved locally')

    # Load and save tokenizer for the model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(args.save_path)
    logger.info('Tokenizer saved locally')
