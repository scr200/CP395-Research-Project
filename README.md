# CP395 Research Project: Predictive vs. Reactive Autoscaling
**Author:** Sufiya Rahemtulla
**Institution:** Wilfrid Laurier University

## Overview
This repository contains the data, source code, and reproducibility artifacts for a hybrid "Predictive + AIOps" cloud autoscaling architecture. The system combines Linear Regression statistical forecasting with a Gemini 2.5 Flash LLM triage agent, safely bounded by deterministic Python guardrails to prevent AI hallucinations.

## Setup & Installation
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/scr200/CP395-Research-Project]
   cd [CP395-Research-Project]

2. **Install dependencies**
   pip install -r requirements.txt

3. **Configure the LLM API Key**
- To execute the AIOps offline triage agent, you must provide your own Google Gemini API Key.
- Create a file named .env in the root directory of this project.
- Add the following line with your key: GEMINI_API_KEY="your_api_key_here"
- Security Note: Ensure that a .gitignore file exists in the root directory containing the line .env to prevent your API key from being uploaded to version          control.

## Repository Structure
- /data : Datasets (raw Google Cluster traces and processed synthetic workloads).
- /src : Core source code, simulation environments, and AIOps agent.
- /experiments : Experiment logs and raw outputs (test_logs.json).
- /figures : Generated plots, graphs, and performance matrices used in the final paper.
- /reports : Final manuscript, literature review, and weekly progress reports.

## Reproducibility Guide
The codebase has been configured using dynamic relative paths (os.path). This ensures that you can run all the following commands directly from the root directory of the repository without modifying any internal file paths, regardless of your operating system.

**Phase 1: Workload Synthesis & Heuristic Baselines**
1. Execute the EDA pipeline to ingest the raw traces and generate the synthetic workload profiles (outputs to the /data folder)
      python src/Initial_Pipeline&EDA.py
2. Run the Baselines & Simulators: Execute the standalone evaluation scripts to simulate the autoscaling environments. This will generate the performance            graphs found in the /figures directory
      python src/baseline_arima.py
      python src/baseline_policy.py
      python src/W6_simulation.py

**Phase 2: AIOps LLM Evaluation**
- Run the LLM Agent: This script parses the 10 anomaly profiles located in test_logs.json, queries the Gemini model for root-cause analysis, and applies the        Python safety guardrails to intercept unsafe scaling recommendations
      python src/aiops_agent.py

**Phase 3: Final Experiment Matrix**
- Evaluate all configurations: This script runs the Reactive baseline, Predictive baseline, and the proposed AIOps policies against the test logs, outputting the final SLA Violation and Resource Waste metrics
   python src/experiment_eval.py

## Notes on Codebase Standards
- **Dynamic Pathing**: Absolute paths and hardcoded strings have been removed. All scripts utilize the os library to dynamically locate the /data and /figures directories.
- **API Security**: The google-genai SDK is wrapped using python-dotenv to ensure LLM credentials remain entirely localized to the user's machine.
