import json
import time
import logging
import subprocess
import pandas as pd
from threading import Thread

# Setup logging
logging.basicConfig(
    format="[%(asctime)s] %(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Global variables to manage state
data_collection = []
stop_collecting = False
collection_thread = None


def _collect_data(shell_command):
    global data_collection, stop_collecting
    logger.info(f"Starting data collection using command: {shell_command}")

    with subprocess.Popen(
        shell_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as proc:
        try:
            while not stop_collecting:
                output = proc.stdout.readline()
                if output == b"" and proc.poll() is not None:
                    break
                if output:
                    json_data = json.loads(output.decode("utf-8"))

                    # Extract system data
                    system_data = json_data.get("system_data", {})
                    vcpu_usage = system_data.get("vcpu_usage", {}).get(
                        "average_usage", {}
                    )
                    memory_info = system_data.get("memory_info", {})

                    memory_total_gb = memory_info.get("memory_total_bytes", 0) / (
                        1024**3
                    )
                    memory_used_gb = memory_info.get("memory_used_bytes", 0) / (1024**3)
                    swap_total_gb = memory_info.get("swap_total_bytes", 0) / (1024**3)
                    swap_used_gb = memory_info.get("swap_used_bytes", 0) / (1024**3)

                    # Extract neuron_runtime_data
                    neuron_runtime_data = json_data.get("neuron_runtime_data", [])
                    for runtime_entry in neuron_runtime_data:
                        report = runtime_entry.get("report", {})
                        neuroncore_counters = report.get("neuroncore_counters", {})
                        neuron_cores_in_use = neuroncore_counters.get(
                            "neuroncores_in_use", {}
                        )

                        neuron_utilization = {}
                        total_utilization = 0
                        neuroncore_count = 0

                        # Calculate the total utilization and count the number of NeuronCores
                        for core_index, core_data in neuron_cores_in_use.items():
                            utilization = core_data.get("neuroncore_utilization", 0)
                            neuron_utilization[
                                f"neuroncore_{core_index}_utilization"
                            ] = utilization
                            neuron_utilization[f"neuroncore_{core_index}_flops"] = (
                                core_data.get("flops", None)
                            )

                            if utilization is not None:
                                total_utilization += utilization
                                neuroncore_count += 1

                        # Calculate mean utilization as a percentage of total possible utilization
                        if neuroncore_count > 0:
                            mean_utilization = total_utilization / neuroncore_count
                        else:
                            mean_utilization = 0

                        flattened_data = {
                            "vcpu_user": vcpu_usage.get("user"),
                            "vcpu_system": vcpu_usage.get("system"),
                            "vcpu_idle": vcpu_usage.get("idle"),
                            "memory_total_gb": memory_total_gb,
                            "memory_used_gb": memory_used_gb,
                            "swap_total_gb": swap_total_gb,
                            "swap_used_gb": swap_used_gb,
                            "neuroncore_count": neuroncore_count,
                            "total_neuroncore_utilization": total_utilization,
                            "mean_neuroncore_utilization_percentage": mean_utilization,
                            "mean_neuroncore_utilization_total_possible": mean_utilization
                            * neuroncore_count,  # out of 100% * number of cores
                        }

                        # Merge neuron utilization data into the flattened_data
                        flattened_data.update(neuron_utilization)

                        # Append the flattened data to the collection
                        data_collection.append(flattened_data)
        except Exception as e:
            logger.error(f"Error occurred during data collection: {e}")
        finally:
            proc.terminate()
            logger.info("Data collection process terminated.")


def start_collection(shell_command):
    """Starts the data collection in a separate thread."""
    global stop_collecting, collection_thread
    stop_collecting = False
    collection_thread = Thread(target=_collect_data, args=(shell_command,))
    collection_thread.start()
    logger.info("Data collection started in the background.")


def stop_collection():
    """Stops the data collection."""
    global stop_collecting, collection_thread
    stop_collecting = True
    if collection_thread:
        collection_thread.join()
    logger.info("Data collection stopped.")


def get_collected_data():
    """Returns the collected data as a Pandas DataFrame."""
    global data_collection
    logger.info(f"Returning collected data with {len(data_collection)} records.")
    return pd.DataFrame(data_collection)


def reset_collection():
    """Resets the collected data and state."""
    global data_collection, stop_collecting, collection_thread
    data_collection = []
    stop_collecting = False
    collection_thread = None
    logger.info("Data collection has been reset.")
