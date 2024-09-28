"""
Collects and logs GPU and CPU metrics to a CSV file.
"""


# Need to install these 2 dependencies:
# pip install psutil==5.9.8
# pip install nvitop==1.3.2

import csv
import time
import psutil
import logging
import tempfile
import pandas as pd
from nvitop import Device, ResourceMetricCollector


# Setup logging
logging.basicConfig(
    format="[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Use a temporary file
temp_file = tempfile.NamedTemporaryFile(mode='w+t', delete=False, suffix='.csv')
csv_file_name = temp_file.name

# Global flag to control data collection for now, change in future.
collecting = True
collected_data = []

def stop_collect(collector=None):
    """
    Stops the data collection process by setting the global flag 'collecting' to False.
    """
    global collecting
    collecting = False
    logger.info("Stopped collection")
    # Calculate GPUMemoryUtilization
    df = pd.DataFrame(collected_data, columns=[
        "timestamp",
        "CPUUtilization",
        "MemoryUtilization",
        "memory_used_mean",
        "DiskUtilization",
        "GPUUtilization",
        "gpu_memory_used_mean",
        "gpu_memory_free_mean",
        "gpu_memory_total_mean",
    ])
    
    # Calculate GPUMemoryUtilization
    df['GPUMemoryUtilization'] = df['gpu_memory_used_mean'] / df['gpu_memory_total_mean'] * 100
    
    # Select and rename the required columns
    return df[["timestamp", "CPUUtilization", "MemoryUtilization", "DiskUtilization", "GPUUtilization", "GPUMemoryUtilization"]]



def _collect_ec2_utilization_metrics():
    """
    Starts the data collection process by initializing the ResourceMetricCollector and collecting metrics at regular intervals.
    """
    global collecting
    logger.info("Starting collection")

    def on_collect(metrics):
        """
        Collects GPU and CPU metrics, then appends them to the CSV file.

        Parameters:
        - metrics: The collected metrics from the ResourceMetricCollector.

        Returns:
        - bool: Returns False if the collection is stopped, otherwise True.
        """
        if not collecting:
            return False

        try:

                # Collect CPU mean utilization
                CPUUtilization = metrics.get(
                    "metrics-daemon/host/cpu_percent (%)/mean", psutil.cpu_percent()
                )
                memory_percent_mean = metrics.get(
                    "metrics-daemon/host/memory_percent (%)/mean",
                    psutil.virtual_memory().percent,
                )
                memory_used_mean = metrics.get(
                    "metrics-daemon/host/memory_used (GiB)/mean",
                    psutil.virtual_memory().available,
                )
                # Disk utilization using host_disk_usage_percent
                # disk_usage_percent = metrics.get(
                #     "metrics-daemon/host/disk_usage_percent (%)/mean", None
                # )

                # Extract the current timestamp
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                # Initialize variables to sum GPU metrics
                total_gpu_utilization = 0
                total_gpu_memory_used = 0
                total_gpu_memory_free = 0
                total_gpu_memory_total = 0

                num_gpus = len(Device.cuda.all())
                # logger.info(f"Number of GPUs detected: {num_gpus}")

                # Iterate over all detected GPUs
                for gpu_id in range(num_gpus):
                    gpu_utilization_mean = metrics.get(
                        f"metrics-daemon/cuda:{gpu_id} (gpu:{gpu_id})/gpu_utilization (%)/mean",
                        None,
                    )
                    gpu_memory_used_mean = metrics.get(
                        f"metrics-daemon/cuda:{gpu_id} (gpu:{gpu_id})/memory_used (MiB)/mean",
                        None,
                    )
                    gpu_memory_free_mean = metrics.get(
                        f"metrics-daemon/cuda:{gpu_id} (gpu:{gpu_id})/memory_free (MiB)/mean",
                        None,
                    )
                    gpu_memory_total_mean = metrics.get(
                        f"metrics-daemon/cuda:{gpu_id} (gpu:{gpu_id})/memory_total (MiB)/mean",
                        None,
                    )

                    if gpu_utilization_mean is not None:
                        total_gpu_utilization += gpu_utilization_mean
                    if gpu_memory_used_mean is not None:
                        total_gpu_memory_used += gpu_memory_used_mean
                    if gpu_memory_free_mean is not None:
                        total_gpu_memory_free += gpu_memory_free_mean
                    if gpu_memory_total_mean is not None:
                        total_gpu_memory_total += gpu_memory_total_mean

                # Calculate the mean values across all GPUs
                GPUUtilization = ( 
                    total_gpu_utilization / num_gpus if num_gpus > 0 else 0
                )
                gpu_utilization_mean_total = (
                    total_gpu_utilization / num_gpus if num_gpus > 0 else None
                )
                gpu_memory_used_mean_total = (
                    total_gpu_memory_used / num_gpus if num_gpus > 0 else None
                )
                gpu_memory_free_mean_total = (
                    total_gpu_memory_free / num_gpus if num_gpus > 0 else None
                )
                gpu_memory_total_mean_total = (
                    total_gpu_memory_total / num_gpus if num_gpus > 0 else None
                )

                # Write the row to the CSV file
                row = [
                    timestamp,
                    CPUUtilization,
                    memory_percent_mean,  # This will be renamed to MemoryUtilization
                    memory_used_mean,
                    psutil.disk_usage('/').percent,  # Add DiskUtilization
                    gpu_utilization_mean_total,
                    gpu_memory_used_mean_total,
                    gpu_memory_free_mean_total,
                    gpu_memory_total_mean_total,
                ]
                # logger.info(f"Writing row: {row}")
                #csv_writer.writerow(row)
                collected_data.append(row)

                with open(csv_file_name, mode="a", newline="") as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(row)                

        except ValueError as e:
            return False

        return True

    # Start the collector and run in the background
    collector = ResourceMetricCollector(Device.cuda.all())
    logger.info("Starting Daemon collector to run in background")
    collector.daemonize(
        on_collect,
        interval=5,
        on_stop=stop_collect,  # Adjust the interval as needed in seconds
    )


def collect_ec2_metrics():
    """
    Initializes the CSV file with headers and starts the metrics collection process.
    """
    global collecting
    collecting = True
    collected_data = []
    # Initialize the CSV file and write the header once
    with open(csv_file_name, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        header = [
            "timestamp",
            "CPUUtilization",
            "MemoryUtilization",
            "memory_used_mean",
            "DiskUtilization",
            "GPUUtilization",
            "gpu_memory_used_mean",
            "gpu_memory_free_mean",
            "gpu_memory_total_mean",
        ]
        logger.info(f"Writing header: {header}")
        csv_writer.writerow(header)

    # Call the function to start collecting metrics
    _collect_ec2_utilization_metrics()
