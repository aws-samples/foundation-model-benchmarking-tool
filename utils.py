import yaml
import logging
import unicodedata
from globals import *
from typing import Dict
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# globals
_tokenizer = None

# utility functions
def load_config(config_file) -> Dict:
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)
    
# The files in LongBench contain nonstandard or irregular Unicode.
# For compatibility and safety we normalize them.

def _normalize(text, form='NFC'):
    return unicodedata.normalize(form, text)


def count_tokens(text: str) -> int:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    return len(_tokenizer.encode(text))

def process_item(item, prompt_fmt: str) -> Dict:
    question = _normalize(item.input)
    context = _normalize(item.context)
    prompt = prompt_fmt.format(question=question, context=context)
    prompt_len = count_tokens(prompt)
    return {
        "question": question,
        "context": context,
        "prompt": prompt,
        "prompt_len": prompt_len,
        "question_len": len(_tokenizer.encode(question)),
        "context_len": len(_tokenizer.encode(context)),
    }
