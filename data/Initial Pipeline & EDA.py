import pandas as pd
import gzip
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

USAGE_FILE = '/Users/sufiya/my_data_folder/classes/CP395/data/instance_usage-000000000000.json.gz'
OUTPUT_CSV = '/Users/sufiya/my_data_folder/classes/CP395/data/week03_cleaned_data.csv' 

# DATA INGESTION PIPELINE 
def load_and_filter_usage(filename, max_rows=500000):
    data = []
    print(f"Reading from: {filename}")
    
    with gzip.open(filename, 'rt') as f:
        for i, line in enumerate(f):
            if i >= max_rows: break
            try:
                row = json.loads(line)

                processed = {
                    'timestamp': int(row['start_time']),
                    'job_id': str(row['collection_id']),
                    'cpu_usage': float(row['average_usage']['cpus']),
                    'mem_usage': float(row['average_usage']['memory'])
                }
                data.append(processed)
            except Exception:
                continue
    return pd.DataFrame(data)

print("Step 1: Ingesting Data...")
df_raw = load_and_filter_usage(USAGE_FILE)

# filter for the most active job to analyze a specific workload
top_job_id = df_raw['job_id'].value_counts().idxmax()
print(f"Analyzing Dominant Job ID: {top_job_id}")
df_job = df_raw[df_raw['job_id'] == top_job_id].copy()

# CLEANING & NORMALIZATION
print("Step 2: Cleaning and Resampling...")
df_job['time'] = pd.to_datetime(df_job['timestamp'], unit='us')
df_job.set_index('time', inplace=True)

# resample to 5 minutes 
df_5min = df_job['cpu_usage'].resample('5min').sum().fillna(0)

# SAVE Intermediate Dataset 
df_5min.to_csv(OUTPUT_CSV)
print(f"Cleaned data saved to {OUTPUT_CSV}")

# REPRODUCIBLE EDA 

# Metric Calculation for Workload Characterization
peak_load = df_5min.max()
mean_load = df_5min.mean()
peak_to_mean_ratio = peak_load / mean_load
print(f"\n--- Workload Statistics ---")
print(f"Peak CPU: {peak_load:.4f}")
print(f"Mean CPU: {mean_load:.4f}")
print(f"Peak-to-Mean Ratio: {peak_to_mean_ratio:.2f} (High ratio = Bursty)")

# FIGURE 1: Time Series 
plt.figure(figsize=(12, 6))
plt.plot(df_5min.index, df_5min.values, label=f'Job {top_job_id}', color='#1f77b4', linewidth=1.5)
plt.title(f'Figure 1: Workload Dynamics (5-Min Aggregation)\nPeak-to-Mean Ratio: {peak_to_mean_ratio:.2f}', fontsize=14)
plt.ylabel('Normalized Compute Units (NCU)', fontsize=12)
plt.xlabel('Time', fontsize=12)

# Fix Squished X-Axis
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M')) # Format: Month-Day Hour:Min
plt.gcf().autofmt_xdate() # Rotates labels nicely

plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('Figure1_TimeSeries.png') # Saves automatically
print("Figure 1 saved as 'Figure1_TimeSeries.png'")
plt.show() # NOTE: close this window to see the next figure

# FIGURE 2: Distribution Analysis
plt.figure(figsize=(10, 6))
sns.histplot(df_5min.values, bins=40, kde=True, color='orange', stat='density')
plt.title('Figure 2: Probability Distribution of CPU Usage\n(Evidence of Heavy-Tail)', fontsize=14)
plt.xlabel('CPU Usage (NCU)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('Figure2_Distribution.png')
print("Figure 2 saved as 'Figure2_Distribution.png'")
plt.show()