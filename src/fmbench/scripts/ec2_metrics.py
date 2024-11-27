"""
Collects and logs GPU and CPU metrics to an EC2 metrics CSV file. This file is populated during the
duration of the inferences against the model deployed on the EC2 instance.
"""
import csv
import time
import psutil
import logging
from fmbench.scripts import constants
from nvitop import Device, ResourceMetricCollector

# Setup logging
logging.basicConfig(
    format="[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Global flag to control data collection for now, change in future.
collecting = True


def stop_collect(collector=None):
    """
    Stops the data collection process by setting the global flag 'collecting' to False.
    """
    global collecting
    collecting = False
    logger.info("Stopped collection")


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
            # Open the CSV file in append mode and write the collected metrics
            with open(constants.EC2_SYSTEM_METRICS_FNAME, mode="a", newline="") as csv_file:
                csv_writer = csv.writer(csv_file)

                # Collect CPU mean utilization
                cpu_percent_mean = metrics.get(
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
                    cpu_percent_mean,
                    memory_percent_mean,
                    memory_used_mean,
                    gpu_utilization_mean_total,
                    gpu_memory_used_mean_total,
                    gpu_memory_free_mean_total,
                    gpu_memory_total_mean_total,
                ]
                # logger.info(f"Writing row: {row}")
                csv_writer.writerow(row)

        except ValueError as e:
            return False

        return True

    # Start the collector and run in the background
    collector = ResourceMetricCollector(Device.cuda.all())
    logger.info("Starting Daemon collector to run in background")
    collector.daemonize(
        on_collect,
        interval=constants.EC2_UTILIZATION_METRICS_INTERVAL,
        on_stop=stop_collect,
    )


def collect_ec2_metrics():
    """
    Initializes the CSV file with headers and starts the metrics collection process.
    """
    global collecting
    collecting = True
    # Initialize the CSV file and write the header once
    with open(constants.EC2_SYSTEM_METRICS_FNAME, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        header = [
            "timestamp",
            "cpu_percent_mean",
            "memory_percent_mean",
            "memory_used_mean",
            "gpu_utilization_mean",
            "gpu_memory_used_mean",
            "gpu_memory_free_mean",
            "gpu_memory_total_mean",
        ]
        logger.info(f"Writing header: {header}")
        csv_writer.writerow(header)

    # Call the function to start collecting metrics
    _collect_ec2_utilization_metrics()
