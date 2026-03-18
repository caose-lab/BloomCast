"""
Usage:
    python3 compute_daily_avg.py --start-date YYYY-MM-DD [--end-date YYYY-MM-DD]
Example:
    python3 compute_daily_avg.py --start-date 2016-01-01 --end-date 2016-01-31
    python3 compute_daily_avg.py --start-date 2016-01-01  # processes until yesterday
"""

import os
from datetime import datetime, timedelta
import argparse
import pandas as pd

par_dir = '/Volumes/T7/data/epa-habs/GOES'
daily_dir = os.path.join(par_dir,'daily_PRWEB_RESULTS_solrad')
# out_dir = os.path.join(par_dir,'averaged_PRWEB_RESULTS_solrad')
out_dir = '/Users/cronjobs/src/BloomCast/src/pipeline/data/goes_data'
# ---------------- USER INPUT ----------------
parser = argparse.ArgumentParser(description="Download and process solar radiation CSVs for SAN_JOSE_LAKE.")
parser.add_argument("--start-date", "-s", required=True, help="Start date (YYYY-MM-DD)")
parser.add_argument("--end-date", "-e", help="End date (YYYY-MM-DD). If omitted, processes from start date until yesterday.")
args = parser.parse_args()

start_date_str = args.start_date
start_date = datetime.strptime(start_date_str, "%Y-%m-%d")

if args.end_date:
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
else:
    end_date = datetime.now() - timedelta(days=1)

results = []
for filename in os.listdir(daily_dir):
    if filename.endswith('.csv'):
        try:
            file_date = datetime.strptime(filename.replace('.csv', ''), "%Y-%m-%d")
        except ValueError:
            continue  # skip files not matching the date format
        if start_date <= file_date <= end_date:
            file_path = os.path.join(daily_dir, filename)
            df = pd.read_csv(file_path)
            magnitude_mean = df['Magnitude'].mean()
            results.append({'date': file_date.strftime('%Y-%m-%d'), 'Magnitude': magnitude_mean})

# Read existing averaged_radiation.csv if it exists
import_path = os.path.join(out_dir, 'averaged_radiation.csv')
if os.path.exists(import_path):
    existing_df = pd.read_csv(import_path)
    # Merge: update existing rows, append new ones
    new_df = pd.DataFrame(results)
    merged_df = pd.concat([existing_df[~existing_df['date'].isin(new_df['date'])], new_df], ignore_index=True)
    merged_df = merged_df.sort_values('date')
    merged_df.to_csv(import_path, index=False)
    print('Updated averaged_radiation.csv')
else:
    # First run: just write new results
    new_df = pd.DataFrame(results)
    new_df = new_df.sort_values('date')
    new_df.to_csv(import_path, index=False)
    print('Created averaged_radiation.csv (first run)')
