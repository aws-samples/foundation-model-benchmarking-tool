from fmbench.utils import *


def process_item(item, prompt_fmt: str) -> Dict:
    question = normalize(item.input)
    context = normalize(item.context)
    prompt = prompt_fmt.format(question=question, context=context)
    prompt_len = count_tokens(prompt)
    ## generalize this further...
    ## bring your own script (if different) - bring your count token and your script
    return {
        "question": question,
        "context": context,
        "prompt": prompt,
        "prompt_len": prompt_len,
        "question_len": tokenizer.count_tokens(question),
        "context_len": tokenizer.count_tokens(context),
    }