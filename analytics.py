"""
Analyze data across multiple fmbench runs
"""
import os
import math
import glob
import json
import yaml
import logging
import argparse
import pandas as pd
from pathlib import Path

logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

RPM_LIST = [1, 10, 100, 1000, 10000]
LATENCY_THRESHOLD: int = 2
RESULTS_DIR: str = "./"
PAYLOAD_FILE_OF_INTEREST: str = "payload_en_3000-3840.jsonl"
MODEL: str = "llama3-8b-instruct"
PRICING_FILE_PATH: str = os.path.join("src", "fmbench", "configs", "pricing.yml")

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
    parser.add_argument('--payload-file',
                        type=str,
                        default=PAYLOAD_FILE_OF_INTEREST,
                        help=f'Payload file representing payload of interest, default={PAYLOAD_FILE_OF_INTEREST}',
                        required=False)
    parser.add_argument('--model-id',
                        type=str,
                        default=MODEL,
                        help=f'Model for which data is being analyzed, default={MODEL}',
                        required=False)
    args = parser.parse_args()
    print(f"main, {args} = args")
    
    # load pricing info
    pricing =  yaml.safe_load(Path(PRICING_FILE_PATH).read_text())
    logger.info(f"pricing={json.dumps(pricing, indent=2)}")
    
    # all results file to be parsed
    all_metrics_summary_files = glob.glob(os.path.join(RESULTS_DIR, "results-*",
                                                       "all_metrics_summary.csv"),
                                          recursive=True)
    logger.info(f"found {len(all_metrics_summary_files)} files {all_metrics_summary_files} ")

    # read all results file in a single dataframe
    df = pd.concat(list(map(pd.read_csv, all_metrics_summary_files)))
    logger.info(f"read {len(all_metrics_summary_files)} files in a dataframe of shape {df.shape}")

    # filter to keep only relevant data
    df_selected = df[df.latency_p95 <= args.latency_threshold]
    logger.info(f"after filtering to keep rows with latency_p95 <= {args.latency_threshold}s, df shape {df_selected.shape}")

    # select row with highest concurrency level
    grouping_cols = ["experiment_name", "payload_file", "instance_type", "instance_count"]
    df_summary_all = df_selected.loc[df_selected.groupby(grouping_cols)['concurrency'].transform(max) == df_selected['concurrency']]

    # find price per txn and price per token
    df_summary_all['cost_per_txn'] = df_summary_all.apply(lambda r: cost_per_txn(r, pricing), axis=1)
    df_summary_all['cost_per_1k_tokens'] = df_summary_all.apply(lambda r: cost_per_1k_tokens(r, pricing), axis=1)

    # extrapolate to price per n requests per minue
    for rpm in RPM_LIST:
        col_name = f"instance_count_and_cost_{rpm}_rpm"
        df_summary_all[col_name] = df_summary_all.apply(lambda r: cost_per_n_rpm(r, rpm, pricing), axis=1)

    df_summary_all = df_summary_all.sort_values(by="cost_per_1k_tokens")

    summary_file: str = f"{args.model_id}-summary-p95-latency={args.latency_threshold}s.csv"
    df_summary_all.to_csv(summary_file, index=False)
    logger.info(f"saved df_summary_all dataframe of shape={df_summary_all.shape} in {summary_file}")
    
    summary_file_payload_of_interest: str = f"{args.model_id}-summary-{Path(args.payload_file).stem}-p95-latency={LATENCY_THRESHOLD}s.csv"
    df_summary_payload_of_interest = df_summary_all[df_summary_all.payload_file == args.payload_file]
    df_summary_payload_of_interest = df_summary_payload_of_interest.sort_values(by="cost_per_1k_tokens")

    df_summary_payload_of_interest.to_csv(summary_file_payload_of_interest, index=False)
    logger.info(f"saved df_summary_payload_of_interest dataframe of "\
                f"shape={df_summary_payload_of_interest.shape} in {summary_file_payload_of_interest}")
    logger.info("all done")

    
if __name__ == "__main__":
    main()