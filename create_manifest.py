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
MANIFEST_MD_FILE: str = "manifest.md"

BASE_FILE_LIST: List[str] = ["prompt_template/.keep",
                             "tokenizer/.keep",
                             "llama2_tokenizer/.keep",
                             "llama3_tokenizer/.keep",
                             "llama3_1_tokenizer/.keep",
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

import subprocess
import re

def get_tree_output(directory='.'):
    """Get the output of the `tree` command for a given directory."""
    result = subprocess.run(['tree', '-f', directory], capture_output=True, text=True)
    return result.stdout

def convert_to_markdown_links(tree_output, directory):
    """Convert tree command output to Markdown hyperlinks."""
    lines = tree_output.splitlines()
    markdown_links = []

    # Regex to match file paths by excluding tree structure characters
    path_pattern = re.compile(r'\s*(?:├──|└──|─|│)?\s*(.*)')

    for line in lines:
        match = path_pattern.match(line)
        if match:
            path = match.group(1).strip()
            if path:
                # Format the path as a Markdown hyperlink
                path_link = path.replace('├── ', '')\
                    .replace('└──', '')\
                    .replace('│   ', '')
                path_readable = path.replace(f"{directory}/", '')
                # if the path is adirectory then we dont want to put a link for it
                # because we dont support directory listing, it will just return a broken
                # page
                from pathlib import Path
                if Path(path).suffix == '':
                    markdown_link = f"**{path_readable}**  "
                else:
                    markdown_link = f'[{path_readable}]({path_link})  '
                markdown_links.append(markdown_link)

    # remove the first line, it is just name of the configs directory
    # remove the last line, it is the number of files and directories
    # like: [14 directories, 69 files](14 directories, 69 files)
    return '\n'.join(markdown_links[1:-1])

def create_dir_listing_as_markdown(directory):
    tree_output = get_tree_output(directory)
    markdown_links = convert_to_markdown_links(tree_output, directory)
    preamble = """Here is a listing of the various configuration files available out-of-the-box with `FMBench`. Click on any link to view a file. You can use these files as-is or use them as templates to create a custom configuration file for your use-case of interest.\n\n"""
    Path(MANIFEST_MD_FILE).write_text(preamble + markdown_links)

def create_manifest_file(config_yml_dir):
    config_yml_files = glob.glob(os.path.join(config_yml_dir, "**/*", "*.yml"),
                                recursive=True)
    config_yml_files = [f.replace(os.path.join("src", "fmbench") + "/", "") for f in config_yml_files]
    print(f"there are {len(config_yml_files)} config yml files")

    # append them to the base list
    all_manifest_files = config_yml_files + BASE_FILE_LIST

    # write to manifest.txt
    written: int = Path(MANIFEST_FILE).write_text("\n".join([f for f in all_manifest_files]))
    print(f"written {written} bytes to {MANIFEST_FILE}")

# all .yml files in the config directory need to be appended to the list above
config_yml_dir = os.path.join("src", "fmbench", "configs")
create_manifest_file(config_yml_dir)

# create the directory listing to put on the website
DOCS_DIR: str = "docs"
CONFIG_DIR_FOR_LISTING: str = os.path.join("configs")
cwd = os.getcwd()
os.chdir(DOCS_DIR)
create_dir_listing_as_markdown(CONFIG_DIR_FOR_LISTING)
os.chdir(cwd)
