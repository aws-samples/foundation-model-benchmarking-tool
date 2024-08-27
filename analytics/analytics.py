"""
Analyze data across multiple fmbench runs
"""
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
from sagemaker_cost_rpm_plot import plot_best_cost_instance_heatmap

logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

RPM_LIST = [1, 10, 100, 1000, 10000]
# default values for latency and concurrency thresholds. To configure the summary table
# based on custom thresholds, add them as command line arguments
LATENCY_THRESHOLD: int = 2
CONCURRENCY_THRESHOLD: int = 1
RESULTS_DIR: str = "./"
ANALYTICS_RESULTS_DIR: str = os.path.join("analytics", "results")
os.makedirs(ANALYTICS_RESULTS_DIR, exist_ok=True)
PAYLOAD_FILE_OF_INTEREST: str = "payload_en_1000-2000.jsonl"
PRICING_FILE_PATH: str = os.path.join("src", "fmbench", "configs",
                                      "pricing.yml")
DEFAULT_COST_WEIGHT: float = 0.6

def cost_per_txn(row, pricing):
    txns_per_hour = row['transactions_per_minute'] * 60
    instance_cost_per_hour = pricing['pricing']['instance_based'][row['instance_type']]
    cost_per_txn = round(instance_cost_per_hour / txns_per_hour, 4)
    return cost_per_txn


def cost_per_1k_tokens(row, pricing):
    txns_per_hour = row['transactions_per_minute'] * 60
    tokens_per_hour = (row['prompt_token_count_mean'] + row['completion_token_count_mean']) * txns_per_hour
    instance_cost_per_hour = pricing['pricing']['instance_based'][row['instance_type']]
    cost_per_1k_tokens = round(1000 * (instance_cost_per_hour / tokens_per_hour), 8)
    return cost_per_1k_tokens

# Determine how many instances would be required to run 100 requests/minute,
# 1000 requests/minute, 10000 requests/minute. The idea being that at the 
# low end of the total number of requests/minute smaller instances which provide
# good inference latency at low concurrencies would suffice (said another way, 
# the larger more expensive instances are an overkill at this stage) but as 
# the number of requests/minute increase there comes an inflexion point beyond
# which the number of smaller instances required would be so much that it 
# would be more economical to use fewer instances of the larger more expensive instances.
def cost_per_n_rpm(r, rpm, pricing):
    instance_count_needed = math.ceil(rpm / r['transactions_per_minute'])
    cost = round(instance_count_needed * pricing['pricing']['instance_based'][r['instance_type']], 2)
    return (instance_count_needed, cost)



def main():
    parser = argparse.ArgumentParser(description='Analyze mukltiple FMBench runs')
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
    summary_file_pattern: str = os.path.join(RESULTS_DIR,
                                             f"results-{args.model_id}-*",
                                             "all_metrics_summary.csv")
    all_metrics_summary_files = glob.glob(summary_file_pattern,
                                          recursive=True)
    files_found: int = len(all_metrics_summary_files)
    logger.info(f"found {files_found} files "
                f"{all_metrics_summary_files} ")
    if files_found == 0:
        logger.error(f"no file found using the following pattern={summary_file_pattern}, exiting")
        sys.exit(1)

    # read all results file in a single dataframe
    df = pd.concat(list(map(pd.read_csv, all_metrics_summary_files)))
    logger.info(f"read {len(all_metrics_summary_files)} files in a dataframe "
                f"of shape {df.shape}")

    # filter to keep only relevant data
    logger.info(f"df columns: {df.columns}")
    # filter for the p95 latency threshold and the concurrency threshold
    df_selected = df[(df.latency_p95 <= args.latency_threshold) & (df.concurrency >= args.concurrency_threshold)]
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
                      'latency_p99', 'cost_per_txn', 'completion_token_throughput']
    # filter out the columns as needed and only give the relevant columns in the analysis markdown table
    df_summary_payload_of_interest_trimmed = df_summary_payload_of_interest.drop(columns=cols_to_remove)
    df_summary_payload_of_interest_trimmed.to_csv(summary_file_payload_of_interest, index=False)
    logger.info(f"saved df_summary_payload_of_interest dataframe of "\
                f"shape={df_summary_payload_of_interest_trimmed.shape} in {summary_file_payload_of_interest}")
    final_table_mkdn: str = 'final_analysis_table'
    final_table_mkdn = Tomark.table(df_summary_payload_of_interest_trimmed.to_dict(orient='records'))
    markdown_file = os.path.join(ANALYTICS_RESULTS_DIR, f"{args.model_id}-analysis-{Path(args.payload_file).stem}-p95-latency={args.latency_threshold}s.md")
    with open(markdown_file, 'w') as f:
        f.write(f"# Analysis for {args.model_id}\n\n")
        f.write(f"## Summary for payload: {Path(args.payload_file).stem}\n\n")
        f.write(final_table_mkdn)
    logger.info(f"Saved analysis Markdown to {markdown_file}")
    logger.info("all done")

    # cost RPM plot, the function saves the html to a file
    heatmap_fname: str = os.path.join(ANALYTICS_RESULTS_DIR,
                                      f"{args.model_id}-cost-rpm-heatmap-for-"
                                      f"{Path(args.payload_file).stem}-p95-latency={args.latency_threshold}s.html")
    df1 = pd.read_csv(summary_file_payload_of_interest)
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


if __name__ == "__main__":
    main()