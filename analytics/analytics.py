"""
Analyze data across multiple fmbench runs
"""
import re
import os
import sys
import math
import glob
import json
import yaml
import logging
import argparse
import pandas as pd
from pathlib import Path
from tomark import Tomark
from sagemaker_cost_rpm_plot import plot_best_cost_instance_heatmap, plot_tps_vs_cost

logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

RPM_LIST = [1, 10, 100, 1000, 10000]
# default values for latency and concurrency thresholds. To configure the summary table
# based on custom thresholds, add them as command line arguments
LATENCY_THRESHOLD: int = 2
CONCURRENCY_THRESHOLD: int = 1
ANALYTICS_RESULTS_DIR: str = os.path.join("analytics", "results")
os.makedirs(ANALYTICS_RESULTS_DIR, exist_ok=True)
PAYLOAD_FILE_OF_INTEREST: str = "payload_en_1000-2000.jsonl"
PRICING_FILE_PATH: str = os.path.join("src", "fmbench", "configs",
                                      "pricing.yml")
DEFAULT_COST_WEIGHT: float = 0.6

def cost_per_txn(row, pricing):
    txns_per_hour = row['transactions_per_minute'] * 60
    if pricing['pricing']['instance_based'].get(row['instance_type']) is not None:
        instance_cost_per_hour = pricing['pricing']['instance_based'][row['instance_type']]
        cost_per_txn = round(instance_cost_per_hour / txns_per_hour, 4)
    else:
        input_token_cost = pricing['pricing']['token_based'][row['instance_type']]['input-per-1k-tokens']
        output_token_cost = pricing['pricing']['token_based'][row['instance_type']]['output-per-1k-tokens']
        cost_per_txn = (row['prompt_token_count_mean']/1000) * input_token_cost + \
                       (row['completion_token_count_mean']/1000) * output_token_cost
        cost_per_txn = round(cost_per_txn, 4)
    return cost_per_txn


def cost_per_1k_tokens(row, pricing):
    txns_per_hour = row['transactions_per_minute'] * 60
    tokens_per_hour = (row['prompt_token_count_mean'] + row['completion_token_count_mean']) * txns_per_hour
    if pricing['pricing']['instance_based'].get(row['instance_type']) is not None:
        instance_cost_per_hour = pricing['pricing']['instance_based'][row['instance_type']]
        cost_per_1k_tokens = round(1000 * (instance_cost_per_hour / tokens_per_hour), 8)
    else:
        input_token_cost = pricing['pricing']['token_based'][row['instance_type']]['input-per-1k-tokens']
        output_token_cost = pricing['pricing']['token_based'][row['instance_type']]['output-per-1k-tokens']
        total_tokens = row['prompt_token_count_mean'] + row['completion_token_count_mean']

        cost_per_1k_tokens = (row['prompt_token_count_mean'] / total_tokens) * input_token_cost + \
                             (row['completion_token_count_mean'] / total_tokens) * output_token_cost
        cost_per_1k_tokens = round(cost_per_1k_tokens, 8)
    return cost_per_1k_tokens

def parse_yaml_config(file_path):
    """
    This function parses a yaml file in the results folder (that represents the configuration file that was used to benchmark)
    and extracts the tensor parallel degree, batch size, and the config file name.
    """
    config_file_properties: Optional[Dict] = None
    tensor_parallel_degree: Optional[str] = None
    batch_size: Optional[int] = None
    serving_properties: Optional[str] = None
    model_copies: Optional[str] = None
    image_uri: Optional[str] = None
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            logger.info(f"Loaded the configuration file: {config}")
        experiment_config = config.get('experiments', [])
        if isinstance(experiment_config, list) and len(experiment_config) == 1:
            serving_properties = experiment_config[0].get('serving.properties', None)
            image_uri = experiment_config[0].get('image_uri', None)
            if serving_properties:
                logger.info(f"serving_properties: {serving_properties}.")
                tp_match = re.search(r'option\.tensor_parallel_degree=(\d+)', serving_properties)
                if tp_match:
                    tensor_parallel_degree = str(tp_match.group(1))

                model_copies = experiment_config[0]['inference_spec'].get('model_copies', None)

                bs_match = re.search(r'option\.max_rolling_batch_size=(\d+)', serving_properties)
                if bs_match:
                    batch_size = int(bs_match.group(1))
            else:
                logger.error("No 'serving.properties' found in the experiment configuration.")
        else:
            logger.error(f"Experiment configuration is list or the number of experiments is not 1, num experiments={len(experiment_config)}")
        config_file_properties = dict(config_file=os.path.basename(file_path),
                                      image_uri=image_uri,
                                      tensor_parallel_degree=tensor_parallel_degree,
                                      batch_size=batch_size,
                                      model_copies=model_copies)
        
    except Exception as e:
        logger.error(f"Error parsing the config file {file_path}: {e}")
        config_file_properties = None
    logger.info(f"config_file_properties={config_file_properties}")
    return config_file_properties

# Determine how many instances would be required to run 100 requests/minute,
# 1000 requests/minute, 10000 requests/minute. The idea being that at the 
# low end of the total number of requests/minute smaller instances which provide
# good inference latency at low concurrencies would suffice (said another way, 
# the larger more expensive instances are an overkill at this stage) but as 
# the number of requests/minute increase there comes an inflexion point beyond
# which the number of smaller instances required would be so much that it 
# would be more economical to use fewer instances of the larger more expensive instances.
def cost_per_n_rpm(r, rpm, pricing):
    if pricing['pricing']['instance_based'].get(r['instance_type']):
        instance_count_needed = math.ceil(rpm / r['transactions_per_minute'])
        cost = round(instance_count_needed * pricing['pricing']['instance_based'][r['instance_type']], 2)
    else:
        input_token_cost = pricing['pricing']['token_based'][r['instance_type']]['input-per-1k-tokens']
        output_token_cost = pricing['pricing']['token_based'][r['instance_type']]['output-per-1k-tokens']
        total_tokens = r['prompt_token_count_mean'] + r['completion_token_count_mean']

        cost_per_txn = (r['prompt_token_count_mean']/1000) * input_token_cost + \
                             (r['completion_token_count_mean']/1000) * output_token_cost
        #txn_per_hour = r['transactions_per_minute'] * 60
        txn_per_hour = rpm * 60
        cost = round(cost_per_txn * txn_per_hour, 8)
        instance_count_needed = 1

    return (instance_count_needed, cost)



def main():
    parser = argparse.ArgumentParser(description='Analyze multiple FMBench runs')
    parser.add_argument('--results-dir',
                        type=str,
                        help=f'Root directory containing results-* folders',
                        required=True)

    parser.add_argument('--exclude-pattern',
                        type=str,
                        default=None,
                        help=f'Exclude result folders matching this pattern, default is None',
                        required=False)
                        
    parser.add_argument('--latency-threshold',
                        type=int,
                        default=LATENCY_THRESHOLD,
                        help=f'Latency threshold, runs with p95 above this are not useful, default={LATENCY_THRESHOLD}',
                        required=False)
    parser.add_argument('--concurrency-threshold', 
                        type=int, 
                        default=CONCURRENCY_THRESHOLD, 
                        help=f'Concurrency threshold, runs with the number of concurrent requests handled under this are not useful, default={CONCURRENCY_THRESHOLD}')
    parser.add_argument('--payload-file',
                        type=str,
                        default=PAYLOAD_FILE_OF_INTEREST,
                        help=f'Payload file representing payload of interest, default={PAYLOAD_FILE_OF_INTEREST}',
                        required=False)
    # the model id is a required field. This model_id must match the model_id in your results folders so it is 
    # used during creating the summary table
    parser.add_argument('--model-id',
                        type=str,
                        help=f'Model for which data is being analyzed, this is a required field',
                        required=True)
    parser.add_argument('--cost-weight',
                        type=float,
                        default=DEFAULT_COST_WEIGHT,
                        help=f"Weightage to assign to cost while choosing best instance type, "
                             f"instance count is assigned \"1 - cost weightage\" automatically, "
                             f"default={DEFAULT_COST_WEIGHT}",
                        required=False)
    
    args = parser.parse_args()
    print(f"main, {args} = args")

    # load pricing info
    pricing =  yaml.safe_load(Path(PRICING_FILE_PATH).read_text())
    logger.info(f"pricing={json.dumps(pricing, indent=2)}")

    # all results file to be parsed
    summary_file_pattern: str = os.path.join(args.results_dir,
                                             f"results-{args.model_id}-*",
                                             "all_metrics_summary.csv")
    all_metrics_summary_files = glob.glob(summary_file_pattern,
                                          recursive=True)
    if args.exclude_pattern is not None:
        all_metrics_summary_files = [f for f in all_metrics_summary_files if args.exclude_pattern not in f]
    files_found: int = len(all_metrics_summary_files)
    logger.info(f"found {files_found} files "
                f"{all_metrics_summary_files} ")
    if files_found == 0:
        logger.error(f"no file found using the following pattern={summary_file_pattern}, exiting")
        sys.exit(1)

    # config file that was used to create the benchmarks
    # in each results diredctory there is a .yml file which is the config file
    result_and_configfile_combinations = []
    for f in all_metrics_summary_files:
        d = str(Path(f).parents[0].absolute())
        config_files = glob.glob(os.path.join(d, '*.yml'))
        if len(config_files) == 0:
            logger.error(f"no config file found in {d}")            
        else:
            result_and_configfile_combinations.append((f, config_files[0]))
    combined_data = []

    logger.info(f"there are {len(result_and_configfile_combinations)} result and config file combinations")

    for result_file, config_file in result_and_configfile_combinations: 
        #zip(all_metrics_summary_files, possible_config_files):
        # Read result and configuration files
        logger.info(f"result_file={result_file},\nconfig_file={config_file}")
        result_df = pd.read_csv(result_file)
        config_info = parse_yaml_config(config_file)
        if config_info:
            config_df = pd.DataFrame([config_info])
            logger.info(f"config_df: {config_df}")
            # match the length of result_df to concat the config file vars to the corresponding 
            # results folder
            config_df_repeated = pd.concat([config_df] * len(result_df), ignore_index=True)
            combined_df = pd.concat([result_df, config_df_repeated], axis=1)
            combined_data.append(combined_df)
        else:
            logger.warning(f"No config data found for {config_file}, using result only.")
            combined_data.append(result_df)
    df = pd.concat(combined_data, ignore_index=True)
    logger.info(f"Final dataframe: {df}")

    # filter to keep only relevant data
    logger.info(f"df columns: {df.columns}")
    # filter for the p95 latency threshold and the concurrency threshold

    df_selected = df[(df.latency_p95 <= args.latency_threshold) & (df.concurrency >= args.concurrency_threshold) & (df.error_rate == 0)]
    logger.info(f"after filtering to keep rows with latency_p95 <= ",
                f"{args.latency_threshold}s, concurrency <=",
                f"{args.concurrency_threshold}",
                f"df shape {df_selected.shape}")


    # select row with highest concurrency level
    grouping_cols = ["experiment_name", "payload_file", "instance_type", "instance_count"]
    # adding selected metrics for when the concurrency is the highest and the completion tokens are given out, indicating valid responses
    df_selected = df_selected[df_selected.completion_token_count_mean.notna()]
    logger.info(f"df_selected: {df_selected.completion_token_count_mean}")
    df_summary_all = df_selected.loc[df_selected.groupby(grouping_cols)['concurrency'].transform(max) == df_selected['concurrency']]

    # find price per txn and price per token
    df_summary_all['cost_per_txn'] = df_summary_all.apply(lambda r: cost_per_txn(r, pricing), axis=1)
    df_summary_all['cost_per_1k_tokens'] = df_summary_all.apply(lambda r: cost_per_1k_tokens(r, pricing), axis=1)

    # extrapolate to price per n requests per minue
    for rpm in RPM_LIST:
        col_name = f"instance_count_and_cost_{rpm}_rpm"
        df_summary_all[col_name] = df_summary_all.apply(lambda r: cost_per_n_rpm(r, rpm, pricing), axis=1)

    df_summary_all = df_summary_all.sort_values(by="cost_per_1k_tokens")
    summary_file: str = os.path.join(ANALYTICS_RESULTS_DIR,
                                     f"{args.model_id}-summary-p95-latency={args.latency_threshold}s.csv")
    df_summary_all.to_csv(summary_file, index=False)
    logger.info(f"saved df_summary_all dataframe of shape={df_summary_all.shape} in {summary_file}")
    
    summary_file_payload_of_interest: str = os.path.join(ANALYTICS_RESULTS_DIR,
                                                         f"{args.model_id}-summary-{Path(args.payload_file).stem}-p95-latency={args.latency_threshold}s.csv")
    summary_file_payload_of_interest_raw_metrics: str = os.path.join(ANALYTICS_RESULTS_DIR,
                                                         f"{args.model_id}-summary-{Path(args.payload_file).stem}-p95-latency-concurrency={args.latency_threshold}s-raw.csv")                                       
    df_summary_payload_of_interest = df_summary_all[df_summary_all.payload_file == args.payload_file]
    df_summary_payload_of_interest = df_summary_payload_of_interest.sort_values(by="cost_per_1k_tokens")
    # create a csv file with all the raw metrics
    df_summary_payload_of_interest.to_csv(summary_file_payload_of_interest_raw_metrics, index=False)
    cols_to_remove = ['payload_file', 'instance_count', 'error_rate', 'prompt_token_count_mean', 'prompt_token_throughput', 'completion_token_count_mean', 'latency_p50',
                      'latency_p99', 'completion_token_throughput']
    # filter out the columns as needed and only give the relevant columns in the analysis markdown table
    df_summary_payload_of_interest_trimmed = df_summary_payload_of_interest.drop(columns=cols_to_remove)
    df_summary_payload_of_interest_trimmed_grouped = df_summary_payload_of_interest_trimmed.loc[df_summary_payload_of_interest_trimmed.groupby('instance_type')['concurrency'].idxmax()].reset_index()
    df_summary_payload_of_interest_trimmed_grouped.to_csv(summary_file_payload_of_interest, index=False)
    logger.info("all done")

    # cost RPM plot, the function saves the html to a file
    heatmap_fname: str = os.path.join(ANALYTICS_RESULTS_DIR,
                                      f"{args.model_id}-cost-rpm-heatmap-for-"
                                      f"{Path(args.payload_file).stem}-p95-latency={args.latency_threshold}s.html")
    df1 = pd.read_csv(summary_file_payload_of_interest)
    logger.info(f"df columns: {df1.columns}")
    # if an instance type has multiple entries then keep the one with the least cost per token
    shape_before = df1.shape
    df1 = df1.loc[df1.groupby('instance_type').cost_per_1k_tokens.idxmin()].reset_index(drop=True)
    shape_after = df1.shape
    if shape_before[0] != shape_after[0]:
        logger.warning(f"there were multiple entries for some instance types, kept ones with min per token cost, "
                       f"shape_before={shape_before}, shape_after={shape_after}")
    prompt_spec: str = args.payload_file.split(".")[0]
    subtitle: str = f"Prompt: {prompt_spec} tokens, latency p95 threshold: {args.latency_threshold}s"
    _ = plot_best_cost_instance_heatmap(df1,
                                    heatmap_fname,
                                    args.model_id,
                                    subtitle,
                                    args.cost_weight,
                                    1 - args.cost_weight)
    # save the line chart
    # TPS vs Cost line chart
    tps_vs_cost_fname: str = os.path.join(ANALYTICS_RESULTS_DIR,
                                        f"{args.model_id}-tps-vs-cost-for-"
                                        f"{Path(args.payload_file).stem}-p95-latency={args.latency_threshold}s.html")

    _ = plot_tps_vs_cost(df1,
                        tps_vs_cost_fname,
                        args.model_id,
                        subtitle)


if __name__ == "__main__":
    main()
