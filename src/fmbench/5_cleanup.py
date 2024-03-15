#!/usr/bin/env python
# coding: utf-8

# ## Clean your deployed model endpoint content
# -----
# 
# #### In this notebook, we will parse through the existing endpoint.json to delete all of the existing endpoints once you are done with running your respective benchmarking tests.
# ***If you are with running all of the tests, and want to delete the existing endpoints, run this notebook.***

## Import all necessary libraries
import json
import boto3
import logging
from fmbench.utils import *
from fmbench.globals import *

## Set your logger to display all of the endpoints being cleaned
logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

## Load the config file to extract our endpoint.json file and its respective file path
config = load_config(CONFIG_FILE)

## Refer to the file path for the endpoint
endpoint_info_list = json.loads(get_s3_object(config['aws']['bucket'], ENDPOINT_LIST_PATH))
logger.info(f"found information for {len(endpoint_info_list)} endpoints, bucket={config['aws']['bucket']}, key={ENDPOINT_LIST_PATH}")
logger.info(json.dumps(endpoint_info_list, indent=2))

## initialize a sagemaker client
sm_client = boto3.client("sagemaker")
# Iterate over the endpoint_info_list and mark the items for deletion
for item in endpoint_info_list:   
    
    ## Extract the endpoint name from the deployed model configuration
    ep_name = item['endpoint']["EndpointName"]
    try:
        ## Describe the model endpoint 
        logger.info(f"Going to describing the endpoint -> {ep_name}")
        resp = sm_client.describe_endpoint(EndpointName=ep_name)
        
        ## If the given model endpoint is in service, delete it 
        if resp['EndpointStatus'] == 'InService':
            logger.info(f"going to delete {ep_name}")
            ## deleting the model endpoint
            sm_client.delete_endpoint(EndpointName=ep_name)
            logger.info(f"deleted {ep_name}")
    except Exception as e:
        logger.error(f"error deleting endpoint={ep_name}, exception={e}")



