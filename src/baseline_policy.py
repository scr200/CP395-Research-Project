import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- DYNAMIC FILE PATH LOGIC ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
FILE_PATH = os.path.join(DATA_DIR, 'week03_updated_cleaned_data.csv')

# LOAD DATA 
df = pd.read_csv(FILE_PATH) 
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# Using the Bursty Job for the stress test
bursty_data = df[df['job_type'] == 'Bursty Job']['cpu_cores']

# DEFINE THE HPA SIMULATOR - reactive baseline
def simulate_hpa(workload, up_threshold=0.8, down_threshold=0.5, capacity_per_node=64):
    allocated_resources = []
    current_nodes = 1
    
    # Simulate a 1-step lag (5 mins)
    for current_load in workload:
        current_capacity = current_nodes * capacity_per_node
        utilization = current_load / current_capacity
        
        # Record prev
        allocated_resources.append(current_capacity)
        
        # DECIDE for NEXT timestamp (Reactive Step)
        if utilization > up_threshold:
            current_nodes += 1 # Scale UP
        elif utilization < down_threshold and current_nodes > 1:
            current_nodes -= 1 # Scale DOWN
            
    return pd.Series(allocated_resources, index=workload.index)

# RUN SIMULATION
allocated = simulate_hpa(bursty_data)

# STATS CALCULATIONS
# Under-provisioning (Demand > Allocation):
violations = bursty_data[bursty_data > allocated]
violation_count = len(violations)
total_violation_amount = (violations - allocated[violations.index]).sum()

# Over-provisioning (When Allocation > Demand):
waste = allocated[allocated > bursty_data] - bursty_data[allocated > bursty_data]
total_waste = waste.sum()

print(f"--- HPA BASELINE RESULTS ---")
print(f"Total SLO Violations (Crashes/Under-Provisioning): {violation_count} time steps")
print(f"Total Resources Wasted (Slack/Over-Provisioning): {total_waste:.2f} Core-Minutes")

# VISUALIZATION
plt.figure(figsize=(15, 6))
plt.plot(bursty_data.index, bursty_data, label='Actual Demand', color='red', alpha=0.5, linewidth=1)
plt.plot(allocated.index, allocated, label='HPA Allocation', color='blue', linestyle='--', linewidth=1)
plt.fill_between(bursty_data.index, bursty_data, allocated, where=(bursty_data > allocated), color='red', alpha=0.3, label='SLO Violation')

plt.title(f"Baseline Overview: HPA Failure Over 30 Days\nTotal Violations: {violation_count} | Total Waste: {total_waste:.0f}", fontsize=14)
plt.ylabel("CPU Cores")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Baseline_HPA_FullMonth.png')
plt.show()

# FIGURE B - 24-Hour Zoom 
start_date = '2019-05-15 00:00:00'
end_date = '2019-05-16 00:00:00'

# Slice the data
zoom_demand = bursty_data[start_date:end_date]
zoom_alloc = allocated[start_date:end_date]

plt.figure(figsize=(15, 6))

# Plot Demand 
plt.plot(zoom_demand.index, zoom_demand, label='Actual Demand (Spikes)', color='red', linewidth=2)
# Plot Allocation 
plt.plot(zoom_alloc.index, zoom_alloc, label='HPA Reaction (Step Scaling)', color='blue', linestyle='--', linewidth=2.5)
# Highlight the Reaction Gap
plt.fill_between(zoom_demand.index, zoom_demand, zoom_alloc, where=(zoom_demand > zoom_alloc), color='red', alpha=0.4, label='SLO Violation (Under-provisioning)')

plt.title("Zoom-In Analysis (24 Hours): The Reaction Gap.", fontsize=14)
plt.ylabel("CPU Cores")
plt.xlabel("Time (Hour of Day)")
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

import matplotlib.dates as mdates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.tight_layout()
plt.savefig('Baseline_HPA_Zoomed.png')
plt.show()
