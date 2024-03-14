from fmbench.utils import *

def process_item(item, prompt_fmt: str) -> Dict:
    text = normalize(item.text)
    prompt = prompt_fmt.format(text=text)
    prompt_len = count_tokens(prompt)
    ## generalize this further...
    ## bring your own script (if different) - bring your count token and your script
    return {
        "prompt": prompt,
        "prompt_len": prompt_len,
        "text_len": tokenizer.count_tokens(text),
    }