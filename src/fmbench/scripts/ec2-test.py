import json
import boto3
import logging
import pandas as pd
from datetime import datetime, timedelta  # Ensure this line is included
from sagemaker_metrics import get_endpoint_metrics

if __name__ == "__main__":
    endpoint_name = "your-endpoint-name"  # Replace with your SageMaker endpoint name
    variant_name = "your-variant-name"    # Replace with your variant name (if applicable)
    namespace = "AWS/EC2"           # Replace with 'AWS/EC2' for EC2 instance metrics
    start_time = datetime.utcnow() - timedelta(hours=1)
    end_time = datetime.utcnow()
    period = 60  # in seconds

    # Retrieve metrics
    metrics_df = get_endpoint_metrics(endpoint_name=endpoint_name,
                                      namespace=namespace,
                                      variant_name=variant_name,
                                      start_time=start_time,
                                      end_time=end_time,
                                      period=period)

    # Display or save the retrieved metrics
    print(metrics_df)
    # Optionally, save to a CSV file
    metrics_df.to_csv('metrics.csv', index=False)
