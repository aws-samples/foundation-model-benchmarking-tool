# import the required libraries
import os
import time
import json
import logging
import sagemaker
import subprocess
from pathlib import Path
from typing import Dict, Optional
from fmbench.scripts import constants

# session/account specific variables
sess = sagemaker.session.Session()

# Initialize the platform where this script deploys the model
PLATFORM: str = constants.PLATFORM_EKS

# Define the location of your s3 prefix for model artifacts
region: str =sess._region_name
HF_TOKEN_FNAME: str = os.path.join(os.path.dirname(os.path.realpath(__file__)), "hf_token.txt")

# set a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _init_eks_checks(eks_cluster_name: str):
    """
    This function describes the EKS cluster, and updates the kubeconfig for the cluser 
    before deploying the model
    """
    try:
        # Describe the EKS Cluster
        logger.info("Describing EKS Cluster...")
        describe_command_args = ["aws", "eks", "--region", region, "describe-cluster", "--name", eks_cluster_name]
        describe_result = subprocess.run(describe_command_args, capture_output=True, text=True)

        # Check if the cluster exists in the user account
        if describe_result.returncode != 0:
            logger.error("Error: EKS cluster does not exists. Please run the Terraform script")
            return
        else:
            logger.info("Describe cluster step done, cluster exists!")

        # Update the kubeconfig before deploying the model
        logger.info("Updating the kubeconfig...")
        update_command_args = ["aws", "eks", "--region", region, "update-kubeconfig", "--name", eks_cluster_name]
        update_result = subprocess.run(update_command_args, capture_output=True, text=True)
        logger.info(update_result.stdout)
        # check if the kubeconfig has been updated
        if update_result.returncode != 0:
            logger.error("Error: kubeconfg not updated")
            return
        else:
            logger.info("kubeconfig updated")

        # Check for the nodes available in the cluster
        logger.info("Showing kubectl")
        show_nodes_command_args = ["kubectl", "version"]
        show_nodes_result = subprocess.run(show_nodes_command_args, capture_output=True, text=True)
        if show_nodes_result.returncode != 0:
            logger.error("Error: nodes not shown")
            return
        else:
            logger.info(f"available kubectl version: {show_nodes_result.stdout}")

        # show the nodes
        logger.info("Showing nodes...")
        show_nodes_command_args = ["kubectl", "get", "nodes"]
        show_nodes_result = subprocess.run(show_nodes_command_args, capture_output=True, text=True)
        if show_nodes_result.returncode != 0:
            logger.error("Error: nodes not shown")
            return
        else:
            logger.info(f"Nodes shown: {show_nodes_result.stdout}")
    except Exception as e:
        logger.error(f"Error occurred while updating the kubeconfig: {e}")


def _deploy_ray(manifest_file_name: str, manifest_dir_path: str):
    """
    This function deploys the model using ray with a kubectl apply command
    """
    try:
        # check the path to the ray file within the configs/eks_manifests directory
        # manifest_ray_fpath: str = os.path.join(MANIFESTS_FOLDER, manifest_file_name)
        current_dir: str = os.path.dirname(os.path.realpath(__file__))
        parent_dir: str = os.path.abspath(os.path.join(current_dir, os.pardir))
        manifest_dir_path = os.path.join(parent_dir, manifest_dir_path)
        # Get the absolute path of the manifest file
        manifest_ray_fpath = os.path.join(manifest_dir_path, manifest_file_name)
        logger.info(f"Manifest file absolute path: {manifest_ray_fpath}")
        # HF token required for gated model downloads form HF
        hf_token_file_path = Path(HF_TOKEN_FNAME)
        if hf_token_file_path.is_file() is True:
            logger.info(f"hf_token file path: {hf_token_file_path} is a file")
            hf_token = Path(HF_TOKEN_FNAME).read_text().strip()
        else:
            logger.info(f"hf_token file path: {hf_token_file_path} is not a file")
        os.environ["AWS_REGION"] = region
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        # make sure the file is at the path
        if not os.path.isfile(manifest_ray_fpath):
            logger.error("Error: Ray file not found")
            subprocess.run("pwd")
            return
        logger.info(f"Deploying the model using the {manifest_file_name} manifest file")
        deploy_result = subprocess.run(f"envsubst < {manifest_ray_fpath} | kubectl apply -f -", shell=True, capture_output=True)
        logger.info(deploy_result.stdout)
        if deploy_result.returncode != 0:
            logger.error(f"Error: Ray not deployed: {deploy_result.stderr}")
        else:
            logger.info(f"Ray Serve Initiated: {deploy_result.stdout}")
    except Exception as e:
        logger.error(f"Error occurred while deploying the model: {e}")


def _check_ray_service_status(eks_model_namespace: str):
    """
    After the deployment step begins, this function checks the status of the model deployment
    every 15 seconds for 30 minutes. If the deployment fails, it errors out.
    """
    try:
        # Set time limit
        start_time: float = time.time()
        # 30 minutes
        timeout: int = (30 * 60)

        while (time.time() - start_time) < timeout:
            logger.info("Checking if Ray service is deployed...")
            # Check the status of the services
            check_command_args = ["kubectl", "-n", eks_model_namespace, "get", "services", "-o", "json"]
            check_result = subprocess.run(check_command_args, capture_output=True, text=True)
            # get the svc count
            status_json = json.loads(check_result.stdout)
            status_list = status_json['items']
            svc_count = len(status_list)
            # Check if all services are in the "Active" state
            if (svc_count == 3):
                logger.info("Ray service is deployed")
                break
            else:
                logger.info("Ray service is not fully deployed yet, waiting for 15 seconds...")
                logger.info(f"Current # of svc deployed: {svc_count}")
                time.sleep(15)
        else:
            logger.error("Error: Ray service is not deployed within 20 minutes.")
    except Exception as e:
        logger.error(f"Error occurred while checking the deployment status of the model: {e}")


def _print_ingress(eks_model_namespace: str, inference_url_format: str) -> str:
    """
    This function prints the endpoint url string that is returned once the model gets
    deployed
    """
    endpoint_url: Optional[str] = None
    try:
        get_command_args = ["kubectl", "-n", eks_model_namespace, "get", "ingress"]
        get_endpoint_result = subprocess.run(get_command_args, capture_output=True, text=True)
        if get_endpoint_result.returncode != 0:
            logger.error("Error: Ingress not printed")

        else:
            logger.info("Ingress Info:")
        # get the ELB ID
        get_elb_id_command_args = ["kubectl", "get", "-n", eks_model_namespace, "ingress", "-o", "jsonpath='{.items[*].status.loadBalancer.ingress[0].hostname}'"]
        get_elb_id_result = subprocess.run(get_elb_id_command_args, capture_output=True, text=True)
        elb_id: str = get_elb_id_result.stdout
        ep_api_url: str = 'http://' + elb_id[1:-1]
        # append the inference format to the endpoint url to run inferences against it
        endpoint_url = ep_api_url + inference_url_format
        logger.info(f"Inference URL: {endpoint_url}")
    except Exception as e:
        logger.error(f"Error occurred while returning the endpoint url: {e}")
        endpoint_url = None
    return endpoint_url


def deploy(experiment_config: Dict, role_arn: str) -> Dict:
    """
    This is the deploy function to run all of the deployment steps to deploy the 
    model on EKS and return a dictionary containing the endpoint url, experiment name, 
    instance type and instance count
    """
    eks_endpoint_info: Optional[Dict] = None
    try:
        # first update the kubeconfig
        eks_cluster_name: str = experiment_config['eks']['eks_cluster_name']
        _init_eks_checks(eks_cluster_name)

        # deploy the model
        manifest_file_name: str = experiment_config['eks']['manifest_file']
        manifest_dir_path: str = experiment_config['eks']['manifest_dir']
        _deploy_ray(manifest_file_name, manifest_dir_path)

        # check the status every 15 seconds during deployment
        eks_model_namespace: str = experiment_config['eks']['eks_model_namespace']
        _check_ray_service_status(eks_model_namespace)

        # fetch the endpoint url once the model is deployed
        inference_url_format: str = experiment_config['inference_spec']['inference_url_format']
        endpoint_url: str = _print_ingress(eks_model_namespace, inference_url_format)
        logger.info(f"Deployed endpoint URL: {endpoint_url}")
        eks_endpoint_info = dict(endpoint_name= endpoint_url,
                                 experiment_name=experiment_config['name'],
                                 instance_type=experiment_config['instance_type'],
                                 instance_count=experiment_config['instance_count'], 
                                 deployed=True)
    except Exception as e:
        logger.error(f"Error occured while deploying the model: {e}")
        eks_endpoint_info = None
    return eks_endpoint_info




