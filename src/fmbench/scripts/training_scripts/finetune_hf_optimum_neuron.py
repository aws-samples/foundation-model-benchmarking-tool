import os
from itertools import chain
from functools import partial
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import load_dataset
from optimum.neuron import NeuronTrainer as Trainer
from optimum.neuron import NeuronTrainingArguments as TrainingArguments
from optimum.neuron.distributed import lazy_load_for_parallelism
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    set_seed,
)

# Global remainder dictionary (used for chunking). We'll use `global remainder` inside functions.
remainder = {"input_ids": [], "attention_mask": [], "token_type_ids": []}


def _format_dolly(sample: Dict) -> str:
    """
    Format a single Dolly sample into the instruction/context/answer prompt.
    """
    instruction = f"### Instruction\n{sample['instruction']}"
    context = f"### Context\n{sample['context']}" if len(sample["context"]) > 0 else None
    response = f"### Answer\n{sample['response']}"
    prompt = "\n\n".join([part for part in [instruction, context, response] if part is not None])
    return prompt


def _pack_dataset(dataset, chunk_length: int = 2048):
    """
    Token-chunking function to split tokenized data into fixed-size blocks.
    Uses a global 'remainder' dictionary to track leftover tokens between batches.
    """
    print(f"Chunking dataset into chunks of {chunk_length} tokens.")

    # (Re)initialize the global remainder before we start chunking
    global remainder
    remainder = {"input_ids": [], "attention_mask": [], "token_type_ids": []}

    def chunk(sample, chunk_length=chunk_length):
        global remainder
        # Concatenate all texts and add remainder from previous batch
        concatenated_examples = {
            k: list(chain(*sample[k])) for k in sample.keys()
        }
        concatenated_examples = {
            k: remainder[k] + concatenated_examples[k]
            for k in concatenated_examples.keys()
        }

        # get total number of tokens for this batch
        batch_total_length = len(concatenated_examples[list(sample.keys())[0]])
        batch_chunk_length = 0
        # determine how many tokens can be split into exact chunks
        if batch_total_length >= chunk_length:
            batch_chunk_length = (batch_total_length // chunk_length) * chunk_length

        # Split by chunks of max_len
        result = {
            k: [
                t[i : i + chunk_length]
                for i in range(0, batch_chunk_length, chunk_length)
            ]
            for k, t in concatenated_examples.items()
        }

        # Update the global remainder with leftover tokens
        remainder = {
            k: concatenated_examples[k][batch_chunk_length:]
            for k in concatenated_examples.keys()
        }

        # Prepare labels (language modeling setup)
        result["labels"] = result["input_ids"].copy()
        return result

    # Apply the chunking function across the dataset
    lm_dataset = dataset.map(partial(chunk, chunk_length=chunk_length), batched=True)
    print(f"Total number of samples after chunking: {len(lm_dataset)}")
    return lm_dataset


def _prepare_dataset(tokenizer, dataset, chunk_length: int = 2048):
    """
    Prepare the Dolly dataset:
    1. Format each sample with instruction/context/answer.
    2. Tokenize.
    3. Chunk into fixed-sized blocks.
    """

    def template_dataset(sample):
        sample["text"] = f"{_format_dolly(sample)}{tokenizer.eos_token}"
        return sample

    # 1. Apply prompt template per sample
    dataset = dataset.map(template_dataset, remove_columns=list(dataset.features))

    # 2. Tokenize dataset
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    )

    # 3. Pack (chunk) dataset
    lm_dataset = _pack_dataset(dataset, chunk_length=chunk_length)
    return lm_dataset


@dataclass
class TrainConfig:
    """
    Simple container for training hyperparameters.
    Feel free to add more fields as needed.
    """
    model_id: str = field(
        default="meta-llama/Meta-Llama-3-8B",
        metadata={"help": "Model checkpoint to train from."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Tensor parallel size for neuron parallelism."},
    )
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "Number of training epochs."}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per device during training."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."}
    )
    output_dir: str = field(
        default="./results",
        metadata={"help": "Directory to store the final model."},
    )


def train(experiment_config: Dict, role_arn: Optional[str] = None) -> Dict:
    """
    Fine-tune a Dolly-like model with Optimum Neuron's Trainer.

    Args:
        experiment_config (Dict): A dictionary with training configuration.
          - 'model_id' (str): Hugging Face model checkpoint to start from
          - 'tensor_parallel_size' (int): Neuron tensor parallelization factor
          - 'num_train_epochs' (int): Number of epochs
          - 'per_device_train_batch_size' (int): Train batch size per device
          - 'seed' (int): RNG seed
          - 'output_dir' (str): Directory to save the model
        role_arn (str, optional): Not used in this local example but retained for API compatibility.

    Returns:
        Dict: A dictionary with training metadata and confirmation of success.
    """

    # 1. Parse config into a dataclass
    config = TrainConfig(
        model_id=experiment_config.get("model_id", "meta-llama/Meta-Llama-3-8B"),
        tensor_parallel_size=experiment_config.get("tensor_parallel_size", 1),
        num_train_epochs=experiment_config.get("num_train_epochs", 1),
        per_device_train_batch_size=experiment_config.get("per_device_train_batch_size", 1),
        seed=experiment_config.get("seed", 42),
        output_dir=experiment_config.get("output_dir", "./results"),
    )

    # 2. Build optimum-neuron TrainingArguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        seed=config.seed,
        # Add other relevant Hugging Face TrainingArguments here
        # e.g. logging_steps=100, save_steps=500, etc.
    )

    # 3. Set random seed
    set_seed(training_args.seed)

    # 4. Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 5. Load and preprocess the Dolly dataset
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    dataset = _prepare_dataset(tokenizer, dataset, chunk_length=2048)

    # 6. Load the model with lazy loading for neuron tensor parallel
    with lazy_load_for_parallelism(tensor_parallel_size=config.tensor_parallel_size):
        model = AutoModelForCausalLM.from_pretrained(config.model_id)

    # 7. Create the NeuronTrainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )

    # 8. Train
    trainer.train()

    # 9. Save final model & tokenizer
    trainer.save_model()

    # 10. Return some metadata about training
    return {
        "model_id": config.model_id,
        "output_dir": config.output_dir,
        "tensor_parallel_size": config.tensor_parallel_size,
        "num_train_epochs": config.num_train_epochs,
        "trained": True,
    }
