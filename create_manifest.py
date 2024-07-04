"""
Script to create a manifest.txt file. The manifest.txt file
lists which (public) files should be copied from the source bucket
to the account specific FMBench bucket
"""
import os
import glob
from typing import List
from pathlib import Path

MANIFEST_FILE: str = "manifest.txt"

BASE_FILE_LIST: List[str] = ["prompt_template/.keep",
                             "tokenizer/.keep",
                             "llama2_tokenizer/.keep",
                             "llama3_tokenizer/.keep",
                             "mistral_tokenizer/.keep",
                             "phi_tokenizer/.keep",
                             "scripts/.keep",
                             "source_data/2wikimqa_e.jsonl",
                             "source_data/2wikimqa.jsonl",
                             "source_data/hotpotqa_e.jsonl",
                             "source_data/hotpotqa.jsonl",
                             "source_data/narrativeqa.jsonl",
                             "source_data/triviaqa_e.jsonl",
                             "source_data/triviaqa.jsonl",
                             "source_data/LICENSE.txt",
                             "source_data/THIRD_PARTY_LICENSES.txt"]

# all .yml files in the config directory need to be appended to the list above
config_yml_files = glob.glob(os.path.join("src", "fmbench", "configs", "**/*", "*.yml"),
                             recursive=True)
config_yml_files = [f.replace(os.path.join("src", "fmbench") + "/", "") for f in config_yml_files]
print(f"there are {len(config_yml_files)} config yml files")

# append them to the base list
all_manifest_files = config_yml_files + BASE_FILE_LIST

# write to manifest.txt
written: int = Path(MANIFEST_FILE).write_text("\n".join([f for f in all_manifest_files]))
print(f"written {written} bytes to {MANIFEST_FILE}")