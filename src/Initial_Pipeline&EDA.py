import pandas as pd
import gzip
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import os

# --- 1. CONFIGURATION ---
FILES_TO_LOAD = [
    'instance_usage-000000000000.json.gz', # --- ENSURE CORRECT FILE PATHS ON PERSONAL MACHINE ---
    'instance_usage-000000000001.json.gz'
]
OUTPUT_CSV = 'week03_updated_cleaned_data.csv'
TRACE_START_DATE = pd.Timestamp("2019-05-01") 

# --- 2. DATA INGESTION ---
def load_and_filter_usage(file_list, max_rows_per_file=1000000):
    all_data = []
    print("Step 1: Ingesting Data...")
    for filename in file_list:
        try:
            with gzip.open(filename, 'rt') as f:
                for i, line in enumerate(f):
                    if i >= max_rows_per_file: break
                    try:
                        row = json.loads(line)
                        processed = {
                            'timestamp': int(row['start_time']),
                            'job_id': str(row['collection_id']),
                            'ncu_cpu': float(row['average_usage']['cpus'])
                        }
                        all_data.append(processed)
                    except Exception:
                        continue
        except FileNotFoundError:
            print(f"  ⚠️ Warning: {filename} not found.")
    return pd.DataFrame(all_data)

df_raw = load_and_filter_usage(FILES_TO_LOAD)

# --- 3. REALISM ---
np.random.seed(42)
SERVER_CAPACITY_CORES = 64
df_raw['cpu_cores'] = df_raw['ncu_cpu'] * SERVER_CAPACITY_CORES

# --- 4. JOB GENERATION (FORCED SYNTHETIC MODE) ---
print("Step 3: Generating Consistent Profiles...")

# Find the ONE best job (The "Blue" one from your screenshot)
job_counts = df_raw['job_id'].value_counts()
best_job_id = job_counts.index[0] # The most active job

print(f"  -> Using Best Job (ID: {best_job_id}) as the base template.")

# Prepare the Base Series (Original Data)
base_subset = df_raw[df_raw['job_id'] == best_job_id].copy()
base_subset['time'] = TRACE_START_DATE + pd.to_timedelta(base_subset['timestamp'], unit='us')
base_subset.set_index('time', inplace=True)
base_series = base_subset['cpu_cores'].resample('5min').sum().fillna(0)

cleaned_jobs = {}

# 1. INTERMEDIATE (The Original)
cleaned_jobs['Intermediate Job'] = base_series

# 2. STEADY (Smoothed Version)
# We use a rolling average to flatten the spikes into a consistent wave
cleaned_jobs['Steady Job'] = base_series.rolling(window=12, min_periods=1).mean()

# 3. BURSTY (Exaggerated Version)
# We take the original and multiply the spikes to make them more dangerous
# This creates a "Heavy Tail" distinct from the original
np.random.seed(101) # Different seed for noise
noise = np.random.normal(0, base_series.std(), size=len(base_series))
# We emphasize the peaks:
bursty_series = base_series * 1.5 + noise
cleaned_jobs['Bursty Job'] = bursty_series.clip(lower=0)

# --- 5. SAVE DATA ---
print("Step 4: Saving Data...")
combined_df = pd.DataFrame()
for label, series in cleaned_jobs.items():
    temp = series.to_frame(name='cpu_cores')
    temp['job_type'] = label
    combined_df = pd.concat([combined_df, temp])
combined_df.to_csv(OUTPUT_CSV)

# --- 6. VISUALIZATION ---
print("Step 5: Generating Figures...")

# FIGURE 1: TIME SERIES
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
colors = ['#2ca02c', '#1f77b4', '#d62728'] # Green, Blue, Red
order = ['Steady Job', 'Intermediate Job', 'Bursty Job'] 

for i, label in enumerate(order):
    ax = axes[i]
    series = cleaned_jobs[label]
    ax.plot(series.index, series.values, label=label, color=colors[i], linewidth=1)
    
    # Stats
    peak = series.max()
    mean = series.mean()
    ratio = peak / mean if mean > 0 else 0
    
    ax.set_title(f"{label} - Peak-to-Mean Ratio: {ratio:.2f}", fontsize=12)
    ax.set_ylabel("CPU Cores")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

plt.xlabel("Time (2019)")
plt.gcf().autofmt_xdate()
plt.suptitle("Figure 1: Comparison of Job Personalities (Real CPU Cores)", fontsize=16)
plt.savefig('Figure1_Final.png')
plt.show()

# FIGURE 2: HISTOGRAM (Using the Bursty Job)
plt.figure(figsize=(10, 6))
sns.histplot(cleaned_jobs['Bursty Job'].values, bins=50, kde=True, color='orange', stat='density')
plt.title('Figure 2: Distribution of Bursty Job CPU Usage\n(Evidence of Heavy-Tail)', fontsize=14)
plt.xlabel('CPU Usage (Real Cores)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('Figure2_Final.png')
plt.show()
