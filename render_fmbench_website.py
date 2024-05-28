"""
Create a _quarto.yml file to render all the available reports as a Quarto website.
Use this _quarto.yml file to render a website using the Quarto docker container.
"""

import os
import json
import glob
import yaml
import docker
import logging
import subprocess

logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_FOLDER_PREFIX: str = "results-"
QUARTO_YML_FNAME: str = '_quarto.yml'

# a canned Quarto website YML file, the code below
# will populate the contents array with the results-* folder
QUARTO_YML_CONTENT: str =  {
        'project': {'type': 'website', 
                    'render': ['results-*/report.md'],
                    'resources': [f'{RESULTS_FOLDER_PREFIX}*/*.csv'],
                    'output-dir': 'fmbench-website'
                   },
        'website': {
            'title': 'FMBench - Foundation Model Benchmarking Tool',
            'sidebar': {
                'style': 'docked',
                'contents': []
            },
            'search': {
                'limit': 5,
                'collapse-after': 1,
                'show-item-context': False
            }
        },
        'format': {'html': {'self-contained': True, 'toc': True}},
    }

logger.info(QUARTO_YML_CONTENT)

# read the results folder available
result_folders = sorted(glob.glob(RESULTS_FOLDER_PREFIX + "*/"))
logger.info(f"there are {len(result_folders)} folders, result_folders={result_folders}")


yml_content = QUARTO_YML_CONTENT
# Append result_folders to the contents if they are not already present
for folder in result_folders:
    folder_entry = {'text': folder.replace(RESULTS_FOLDER_PREFIX, ""), 'href': f'{folder}/report.md'}
    folder_entry_splitted = folder.split("-")
    # the second and third tokens constitute the model id
    # for example results-llama3-8b-trn1-32xl-tp=8-bs=4-byoe
    # have llama3-8b as the model
    model_id = f"{folder_entry_splitted[1]}-{folder_entry_splitted[2]}"
    model_id = model_id.lower()
    if yml_content['website']['sidebar']['contents'] != []:
        sections = [e['section'] for e in yml_content['website']['sidebar']['contents'] if e]
    else:
        sections = []
    if model_id not in sections:
        #print(json.dumps(yml_content, indent=2))
        logger.info(f"{model_id} not in {sections}, creating a new section")
        yml_content['website']['sidebar']['contents'] = [{'section': model_id,
                                                         'collapsed': True,
                                                         'contents': [folder_entry]}]
    else:
        logger.info(f"{model_id} is present in {sections}, going to append folder entry to it")
        for e in yml_content['website']['sidebar']['contents']:
            if e['section'] == model_id:
                e['contents'].append(folder_entry)

# Save the modified YAML content back to the file
with open(QUARTO_YML_FNAME, 'w') as yml_fp:
    yaml.dump(yml_content, yml_fp, default_flow_style=False)

logger.info(f"Result folders have been appended to {json.dumps(yml_content, indent=2)}")

# _quarto.yml created, going to render the website note
docker_running = False

try:
    docker_client = docker.DockerClient()
    docker_running = docker_client.ping()
except Exception as e:
    logger.error(f"seems like docker is not installed or not running, exception={e}")

if docker_running is True:
    cmd = f"docker run --rm -v $(pwd):/public -w /public -u $(id -u):$(id -g) registry.gitlab.com/quarto-forge/docker/quarto quarto render"
    logger.info(f"going to create self-conained html wensite with cmd=\"{cmd}\"")
    logger.info("**this will take a minute or so**")
    process = subprocess.Popen(cmd,
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True,
                               shell=True)
    std_out, std_err = process.communicate()
    logger.info(std_out.strip())
    logger.info(std_err)

else:
    logger.error(f"docker is not available, not going to create self-conained html report")