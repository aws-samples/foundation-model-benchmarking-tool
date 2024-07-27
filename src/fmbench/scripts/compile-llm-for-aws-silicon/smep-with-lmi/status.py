import boto3
import os
import sys

aws_region = 'us-east-1'
os.environ['AWS_DEFAULT_REGION'] = aws_region

if __name__ == "__main__":
    ENDPOINT_MAME = sys.argv[1]
    boto3_session=boto3.session.Session(region_name="us-east-1")
    smr = boto3.client('sagemaker-runtime')
    sm = boto3.client('sagemaker')
    response = sm.describe_endpoint(EndpointName=ENDPOINT_MAME)
    print(f"Status of Endpoint {response['EndpointName']}")
    print(f"{response['EndpointStatus']}")