"""
Create a mkdocs.yml file to render all the available reports as a MKdocs website.
"""

import os
import json
import glob
import yaml
import copy
import logging
import argparse
from pathlib import Path

logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


HOME_DIR = str(Path.home())
DEFAULT_RESULTS_ROOT = os.path.join(HOME_DIR, "fmbench_data")
RESULTS_FOLDER_PREFIX: str = "results-"

SCRIPT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

MKDOCS_YML_FNAME: str = os.path.join(SCRIPT_DIR_PATH, 'mkdocs.yml')
MKDOCS_TEMPLATE_YML_FNAME: str = os.path.join(SCRIPT_DIR_PATH, 'mkdocs_template.yml')

def main():
    parser = argparse.ArgumentParser(description='Analyze multiple FMBench runs')
    parser.add_argument('--results-root-dir',
                        type=str,
                        default=DEFAULT_RESULTS_ROOT,
                        help=f'Root directory containing results-* folders, default={DEFAULT_RESULTS_ROOT}',
                        required=False)

    args = parser.parse_args()
    logger.info(f"main, {args} = args")

    # find out the results folders
    result_folders = glob.glob(os.path.join(args.results_root_dir, RESULTS_FOLDER_PREFIX + "*"))
    logger.info(f"there are {len(result_folders)} results folders\n {result_folders}")

    # read the mkdocs template to add the folders to the content
    content = Path(MKDOCS_TEMPLATE_YML_FNAME).read_text()
    mkdocs_template = yaml.safe_load(content)
    logger.info(f"read mkdocs_template=\n{json.dumps(mkdocs_template, indent=2)})")
    mkdocs = copy.deepcopy(mkdocs_template)
    for f in result_folders:
        f = Path(f).name
        logger.info(f"folder={f}")
        # folder is of the form results-llama3-1-8b-inf2.48xl-tp=8-bs=4-mc=1-ec2
        label = f.replace(RESULTS_FOLDER_PREFIX, "")
        tokens = label.split("-")
        model_id = tokens[0]
        found = False
        for e in mkdocs['nav']:
            keys = list(e.keys())
            logger.info(f"keys={keys}")

            if model_id in keys:
                logger.info(f"{model_id} key already exists in element = {json.dumps(e, indent=2)}")
                e[model_id].append({label: os.path.join(f, "report.md")})
                found = True
                break
            else:
                logger.info(f"model_id={model_id} not in keys={keys}")
        if found is False:
            logger.info(f"model_id={model_id} not found in any existing keys, going to create a new key")
            mkdocs['nav'].append({model_id: [{label: os.path.join(f, "report.md")}]})
            logger.info(f"after adding model_id={model_id}, mkdocs=\n{json.dumps(mkdocs, indent=2)})")

    logger.info(f"final mkdocs yml =\n{mkdocs}")
    # write it to the mkdocs.yml file
    with open(MKDOCS_YML_FNAME, 'w') as outfile:
        yaml.dump(mkdocs, outfile, default_flow_style=False)


if __name__ == "__main__":
    main()

