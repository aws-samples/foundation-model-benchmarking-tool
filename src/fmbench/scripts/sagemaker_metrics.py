"""
Retrieves metrics for SageMaker Endpoints from CloudWatch.
See https://docs.aws.amazon.com/sagemaker/latest/dg/monitoring-cloudwatch.html for
full list of metrics.
"""
import json
import boto3
import logging
import pandas as pd
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def _get_endpoint_utilization_metrics(endpoint_name: str,
                                      variant_name: str,
                                      start_time: datetime,
                                      end_time: datetime,
                                      period : int = 60) -> pd.DataFrame:
    """
    Retrieves utilization metrics for a specified SageMaker endpoint within a given time range.

    Parameters:
    - endpoint_name (str): The name of the SageMaker endpoint.
    - start_time (datetime): The start time for the metrics data.
    - end_time (datetime): The end time for the metrics data.
    - period (int): The granularity, in seconds, of the returned data points. Default is 60 seconds.

    Returns:
    - Dataframe: A Dataframe containing metric values for utilization metrics like CPU and GPU Usage.
    """
    
    metrics = ["CPUUtilization",
               "MemoryUtilization",
               "DiskUtilization",
               "InferenceLatency",
               "GPUUtilization",
               "GPUMemoryUtilization"]
    
    client = boto3.client('cloudwatch')
    data = []
    namespace = "/aws/sagemaker/Endpoints"
    
    for metric_name in metrics:
        logger.debug(f"_get_endpoint_utilization_metrics, endpoint_name={endpoint_name}, variant_name={variant_name}, "
                     f"metric_name={metric_name}, start_time={start_time}, end_time={end_time}")
        response = client.get_metric_statistics(
            Namespace=namespace,
            MetricName=metric_name,
            Dimensions=[
                {
                    'Name': 'EndpointName',
                    'Value': endpoint_name
                },
                {
                    'Name': 'VariantName',
                    'Value': variant_name
                }
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=period,
            Statistics=['Average']  # You can also use 'Sum', 'Minimum', 'Maximum', 'SampleCount'
        )
        logger.debug(response)
        for datapoint in response['Datapoints']:
            data.append({
                'EndpointName': endpoint_name, 
                'Timestamp': datapoint['Timestamp'],
                'MetricName': metric_name,
                'Average': datapoint['Average']
            })

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    # Pivot the DataFrame to have metrics as columns
    df_pivot = df.pivot_table(index=['Timestamp', 'EndpointName'], columns='MetricName', values='Average').reset_index()
    
    # Remove the index column heading
    sm_utilization_metrics_df = df_pivot.rename_axis(None, axis=1)
    
    return sm_utilization_metrics_df


def _get_endpoint_invocation_metrics(endpoint_name: str,
                                     variant_name: str,
                                     start_time: datetime,
                                     end_time: datetime,
                                     period : int = 60):
    """
    Retrieves Invocation metrics for a specified SageMaker endpoint within a given time range.

    Parameters:
    - endpoint_name (str): The name of the SageMaker endpoint.
    - start_time (datetime): The start time for the metrics data.
    - end_time (datetime): The end time for the metrics data.
    - period (int): The granularity, in seconds, of the returned data points. Default is 60 seconds.

    Returns:
    - Dataframe: A Dataframe containing metric values for Invocation metrics like Invocations and Model Latency.
    """
    metric_names = ["Invocations",
                    "Invocation4XXErrors",
                    "Invocation5XXErrors",
                    "ModelLatency",
                    "InvocationsPerInstance"]
    
    # Initialize a session using Amazon CloudWatch
    client = boto3.client('cloudwatch')

    namespace = "AWS/SageMaker"
    data = []
    
    for metric_name in metric_names:
        if metric_name == 'ModelLatency':
            stat = 'Average'
        else:
            stat = 'Sum'
        logger.debug(f"_get_endpoint_invocation_metrics, endpoint_name={endpoint_name}, variant_name={variant_name}, "
                     f"metric_name={metric_name}, start_time={start_time}, end_time={end_time}")
        # Get metric data for the specified metric
        response = client.get_metric_data(
            MetricDataQueries=[
                {
                    'Id': f'metric_{metric_name}',
                    'MetricStat': {
                        'Metric': {
                            'Namespace': namespace,
                            'MetricName': metric_name,
                            'Dimensions': [
                                {
                                    'Name': 'EndpointName',
                                    'Value': endpoint_name
                                },
                                {
                                    'Name': 'VariantName',
                                    'Value': variant_name
                                }
                            ]
                        },
                        'Period': period,  # Period in seconds
                        'Stat': stat  # Statistic to retrieve
                    },
                    'ReturnData': True,
                },
            ],
            StartTime=start_time,
            EndTime=end_time
        )
        logger.debug(response)
        # Extract the data points from the response
        timestamps = response['MetricDataResults'][0]['Timestamps']
        values = response['MetricDataResults'][0]['Values']
        
        for timestamp, value in zip(timestamps, values):
            data.append({
                'EndpointName': endpoint_name, 
                'Timestamp': timestamp,
                'MetricName': metric_name,
                'Value': value
            })

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)
    
    # Pivot the DataFrame to have metrics as columns
    df_pivot = df.pivot_table(index=['Timestamp', 'EndpointName'], columns='MetricName', values='Value').reset_index()
    
    # Remove the index column heading
    sm_invocation_metrics_df = df_pivot.rename_axis(None, axis=1)
    
    return sm_invocation_metrics_df


def get_endpoint_metrics(endpoint_name: str,
                         variant_name: str,
                         start_time: datetime,
                         end_time: datetime,
                         period: int = 60):
    """
    Retrieves Invocation and Utilization metrics for a specified SageMaker endpoint within a given time range.

    Parameters:
    - endpoint_name (str): The name of the SageMaker endpoint.
    - start_time (datetime): The start time for the metrics data.
    - end_time (datetime): The end time for the metrics data.
    - period (int): The granularity, in seconds, of the returned data points. Default is 60 seconds.

    Returns:
    - Dataframe: A Dataframe containing metric values for Utilization and Invocation metrics.
    """
    
    endpoint_metrics_df: Optional[pd.DataFrame] = None
    try:
        logger.info(f"get_endpoint_metrics, going to retrieve endpoint utlization metrics for "
                    f"endpoint={endpoint_name}, variant_name={variant_name}, start_time={start_time}, "
                    f"end_time={end_time}, period={period}")
        utilization_metrics_df = _get_endpoint_utilization_metrics(endpoint_name=endpoint_name,
                                                                   variant_name=variant_name,
                                                                   start_time=start_time,
                                                                   end_time=end_time,
                                                                   period=period)
        logger.info(f"get_endpoint_metrics, going to retrieve endpoint invocation metrics for "
                    f"endpoint={endpoint_name}, variant_name={variant_name}, start_time={start_time}, "
                    f"end_time={end_time}, period={period}")
        invocation_metrics_df = _get_endpoint_invocation_metrics(endpoint_name=endpoint_name,
                                                                 variant_name=variant_name,
                                                                 start_time=start_time,
                                                                 end_time=end_time,
                                                                 period=period)

        endpoint_metrics_df = pd.merge(utilization_metrics_df,
                                       invocation_metrics_df,
                                       on=['Timestamp', 'EndpointName'],
                                       how='outer')
        logger.info(f"get_endpoint_metrics, shape of invocation and utilization metrics for "
                    f"endpoint={endpoint_name} is {endpoint_metrics_df.shape}")
        logger.info(f"get_endpoint_metrics, endpoint_metrics_df={endpoint_metrics_df.head()}")
    except Exception as e:
        logger.error(f"get_endpoint_metrics, exception occured while retrieving metrics for {endpoint_name}, "
                     f"exception={e}")

    return endpoint_metrics_df