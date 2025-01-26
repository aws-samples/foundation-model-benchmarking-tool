import os
import sys
import time
import torch
import logging
import argparse
import torch_neuronx
from transformers import AutoTokenizer
from transformers_neuronx.config import GenerationConfig
from transformers_neuronx.llama.model import LlamaForSampling
from transformers_neuronx import LlamaForSampling, NeuronConfig, GQA, QuantizationConfig

root = logging.getLogger()
if root.handlers:
    for handler in root.handlers:
        root.removeHandler(handler)
logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    
    # Define and parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        type=str,
        help="What do you want the script do: \"compile\" or \"infer\"?"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size with which to compile the model, default=4",
    )

    parser.add_argument(
        "--num-neuron-cores",
        type=int,
        default=8,
        help="Number of Neuron cores in the instance in which the model would be run, default=8",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        help="Directory from where to read the model binaries",
    )
    args = parser.parse_args()
    logger.info(f"args={args}")

    # we will pin cores to 8 for inf2.24xlarge 
    os.environ['NEURON_RT_NUM_CORES'] = str(args.num_neuron_cores)
    os.environ["NEURON_CC_FLAGS"] = "-O3"  ## for best perf
    BATCH_SIZE = args.batch_size
    CONTEXT_LENGTH = 44 # hard coded for sample prompt
    model_dir = args.model_dir
    model_compiled_dir = os.path.join(os.path.dirname(os.path.dirname(model_dir)), "neuronx_artifacts")
    neuron_config = NeuronConfig(on_device_embedding=False,
                                attention_layout='BSH',
                                fuse_qkv=True,
                                group_query_attention=GQA.REPLICATED_HEADS,
                                quant=QuantizationConfig(),
                                on_device_generation=GenerationConfig(do_sample=True))

    if args.action == "compile":
        start = time.perf_counter()
        model = LlamaForSampling.from_pretrained(
                model_dir,
                batch_size=args.batch_size,
                tp_degree=args.num_neuron_cores,
                amp='f16',
                neuron_config=neuron_config,
                n_positions=4096,
                )
        model.to_neuron()
        # save model to the disk
        model.save(model_compiled_dir)
        elapsed = time.perf_counter() - start
        logger.info(f'\nCompilation and loading took {elapsed:.2f} seconds\n')