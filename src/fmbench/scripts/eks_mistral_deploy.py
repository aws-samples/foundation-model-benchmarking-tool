import os
import sys
import subprocess
import time
import json
from typing import Dict
# Constants
REGION = "us-west-2"
HUGGING_FACE_HUB_TOKEN = "<hf_token_here>"
CLUSTER_NAME = "trainium-inferentia"
RAY_PATH = "./src/fmbench/scripts/manifests/mistral/mistral-ray-service.yaml"

def init_eks_checks():
    # Describe the EKS Cluster
    print("Describing EKS Cluster...")
    describe_command_args = ["aws", "eks", "--region", REGION, "describe-cluster", "--name", CLUSTER_NAME]
    describe_result = subprocess.run(describe_command_args, capture_output=True, text=True)

    # Check the describe_result
    if describe_result.returncode != 0:
        print("Error: EKS cluster does not exists. Please run the Terraform script")

    else:
        print("Describe cluster step done, cluster exists!")
        # print(describe_result.stdout)

    # Update the kubeconfig
    print("Updating kubeconfig...")
    update_command_args = ["aws", "eks", "--region", REGION, "update-kubeconfig", "--name", CLUSTER_NAME]
    update_result = subprocess.run(update_command_args, capture_output=True, text=True)
    print(update_result.stdout)
    if update_result.returncode != 0:
        print("Error: kubeconfg not updated")

    else: 
        print("kubeconfig updated")
        print(update_result.stdout)


    print("Showing kubectl")
    show_nodes_command_args = ["kubectl", "version"]
    show_nodes_result = subprocess.run(show_nodes_command_args, capture_output=True, text=True)
    if show_nodes_result.returncode != 0:
        print("Error: nodes not shown")

    else:
        print("version shown:")
        print(show_nodes_result.stdout)
    
    # show the nodes
    print("Showing nodes...")
    show_nodes_command_args = ["kubectl", "get", "nodes"]
    show_nodes_result = subprocess.run(show_nodes_command_args, capture_output=True, text=True)
    if show_nodes_result.returncode != 0:
        print("Error: nodes not shown")

    else:
        print("Nodes shown:")
        print(show_nodes_result.stdout)

def deploy_ray():
    print("deploy it")
    os.environ["AWS_REGION"] = REGION
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HUGGING_FACE_HUB_TOKEN
    # print("this is the hf token being used")
    # echo_result = subprocess.run("echo $HUGGING_FACE_HUB_TOKEN", shell=True, capture_output=True)
    # print(echo_result.stdout)
    
    #Command envsubst < ./manifests/mistral/mistral-ray-service.yaml| kubectl apply -f -
    print("new run from manifest file: ", RAY_PATH)
    # make sure the file is at the path
    if not os.path.isfile(RAY_PATH):
        print("Error: Ray file not found")
        subprocess.run("pwd")
    deploy_result = subprocess.run(f"envsubst < {RAY_PATH} | kubectl apply -f -", shell=True, capture_output=True)
    print(deploy_result.stdout)

    if deploy_result.returncode != 0:
        print("Error: Ray not deployed")
        print(deploy_result.stderr)

    else:
        print("Ray Serve Initiated...")
        print(deploy_result.stdout)

    print("check it")
    check_ray_service_status()


def deploy_ingress():
    print("Deploying Ingress...")
    deploy_ingress_command_args = ["kubectl", "apply", "-f", INGRESS_PATH]
    deploy_result = subprocess.run(deploy_ingress_command_args, capture_output=True, text=True)
    print(deploy_result.stdout)
    if deploy_result.returncode != 0:
        print("Error: Ingress not deployed")

    else:
        print("Ingress deployed")
        print(deploy_result.stdout)
    
def check_pod_status():
    # Set time limit
    start_time = time.time()
    timeout = 20 * 60 # 20 minutes

    while time.time() - start_time < timeout:
        print("Checking if Ray is deployed...")
        # standard get
        get_command_args = ["kubectl", "-n", "mistral", "get", "pods"]
        get_result = subprocess.run(get_command_args, capture_output=True, text=True)
        print(get_result.stdout)
        
        # status
        check_command_args = ["kubectl", "-n", "mistral", "get", "pods", "-o", "jsonpath='{.items[*].status.phase}'"]
        check_result = subprocess.run(check_command_args, capture_output=True, text=True)
        status_list = check_result.stdout.strip("\n").strip("'").split(" ")
        
        if all(status == "Running" for status in status_list):
            print("Ray is deployed")
            break
        else:
            print("Ray is not deployed yet, waiting for 15 seconds...")
            print(status_list)
            time.sleep(15)
    else:
        print("Error: Ray is not deployed 15 mins reached.")


def check_ray_service_status():
    # Set time limit
    start_time = time.time()
    timeout = 20 * 60  # 20 minutes

    while time.time() - start_time < timeout:
        print("Checking if Ray service is deployed...")
        
        # Check the status of the services
        check_command_args = ["kubectl", "-n", "mistral", "get", "services", "-o", "json"]
        check_result = subprocess.run(check_command_args, capture_output=True, text=True)
        
        # get the count of svc
        status_json = json.loads(check_result.stdout)
        status_list = status_json['items']
        svc_count = len(status_list)

        # Check if all services are in the "Active" state
        if (svc_count == 3):
            print("Ray service is deployed")
            break
        else:
            print("Ray service is not fully deployed yet, waiting for 15 seconds...")
            print("Current # of svc deployed: ",svc_count)
            time.sleep(15)
    else:
        print("Error: Ray service is not deployed within 20 minutes.")


def print_ingress():
    get_command_args = ["kubectl", "-n", "mistral", "get", "ingress"]
    get_result = subprocess.run(get_command_args, capture_output=True, text=True)
    if get_result.returncode != 0:
        print("Error: Ingress not printed")

    else:
        print("Ingress Info:")
        # print(get_result.stdout)
    # get the ELB ID
    get_elb_id_command_args = ["kubectl", "get", "-n", "mistral", "ingress", "-o", "jsonpath='{.items[*].status.loadBalancer.ingress[0].hostname}'"]
    get_elb_id_result = subprocess.run(get_elb_id_command_args, capture_output=True, text=True)
    ELB_ID = get_elb_id_result.stdout
    
    
    # base URL
    HTTPS = 'http://'
    API_URL = HTTPS + ELB_ID[1:-1]
    
    base_url = API_URL + '/mistral/'

    dashboard_url = API_URL + '/mistral/dashboard/'
    print("Ray Dashboard URL: " + dashboard_url)
    
    inference_url = API_URL + '/mistral/serve/infer?sentence='
    print("Inference URL: " + inference_url)
    
    base_url = API_URL + '/mistral/serve/infer?'
    print("Inference URL: " + inference_url)
    
    return base_url

def clean_up():
    print("Cleaning up...")
    delete_command_args = ["kubectl", "delete", "-f", RAY_PATH]
    delete_result = subprocess.run(delete_command_args, capture_output=True, text=True)
    print(delete_result.stdout)

def main():
    print("init. eks...")
    init_eks_checks()
    
    print("Deploying Ray")
    deploy_ray()
    
    print("check service status")
    check_ray_service_status()
    print("printing ingress")
    return print_ingress()


def deploy(experiment_config: Dict, role_arn: str) -> Dict:
    print("Deploying...")
    endpoint_name_local = main()
    print("here is the ep:", endpoint_name_local)
    return dict(endpoint_name= endpoint_name_local, experiment_name=experiment_config['name'])




