# CP395 Research Project

## Setup
1. Clone the repository
2. Install dependencies:
   pip install -r requirements.txt

## Repository Structure
- data/        : Datasets (raw and processed)
- src/         : Source code
- experiments/ : Experiment scripts and notebooks
- figures/     : Generated plots and figures
- reports/     : Papers, reports, and notes

# Reproducibility

To reproduce this project, first install the required dependencies by running `pip install -r requirements.txt`. 

Next, execute `Initial Pipeline & EDA.py` to ingest the raw Google Cluster traces and generate the synthetic workload profiles (producing the `week03_updated_cleaned_data.csv` file). 

Finally, you can run any of the standalone evaluation scripts, such as `W6 simulation.py`, `baseline_arima.py`, or `baseline_policy.py`, to simulate the autoscaling environments and generate the output graphs.

**Ensure that the file path for on your machine is updatedin the script (`week03_updated_cleaned_data.csv') prior to running**
