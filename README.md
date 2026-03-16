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
## Reproducibility

### Phase 1: Workload Synthesis & Heuristic Baselines (Weeks 1-6)
To reproduce the initial data processing and heuristic baseline evaluations:

1. Execute `Initial Pipeline & EDA.py` to ingest the raw Google Cluster traces and generate the synthetic workload profiles (producing the `week03_updated_cleaned_data.csv` file). 
   * **Note:** Ensure that the file path for `week03_updated_cleaned_data.csv` on your local machine is updated in the script prior to running.
2. Run any of the standalone evaluation scripts, such as `W6 simulation.py`, `baseline_arima.py`, or `baseline_policy.py`, to simulate the autoscaling environments and generate the output graphs.

### Phase 2: AIOps LLM Evaluation (Weeks 7-9)
To reproduce the LLM Anomaly Triage Agent evaluations and the resulting Experiment Matrix:

1. **Run the LLM Agent (Week 7):**
   This script parses the 10 anomaly profiles located in `test_logs.json`, queries the Gemini 2.5 Flash model for diagnostic reasoning, and applies the Python safety guardrails to intercept unsafe scaling recommendations.
   `python src/aiops_agent.py`

2. **Generate the Experiment Matrix (Week 8):**
   This script aggregates the performance of the anomaly logs to calculate SLA violations, resource waste, and latency metrics. It compares the Reactive and Predictive baseline configurations against the proposed AIOps system and runs the guardrail ablation study.
   `python src/experiment_eval.py`
