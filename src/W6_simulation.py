# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import gym
from gym import spaces


# REPRODUCIBILITY
SEED = 42
np.random.seed(SEED)

# 2. PREPARE DATA & INTEGRATE PREDICTOR (Linear Regression)

print("Loading data and generating Linear Regression forecasts...")
df = pd.read_csv('week03_updated_cleaned_data.csv') # --- ENSURE CORRECT FILE PATH ON PERSONAL MACHINE
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# Filter for the "Bursty Job"
data = df[df['job_type'] == 'Bursty Job']['cpu_cores']

# Feature Engineering (lag features from Week 5)
ml_data = pd.DataFrame(data)
ml_data['Lag_1'] = ml_data['cpu_cores'].shift(1)
ml_data['Lag_2'] = ml_data['cpu_cores'].shift(2)
ml_data['Lag_3'] = ml_data['cpu_cores'].shift(3)
ml_data.dropna(inplace=True)

# Train the winning LR model on the whole dataset to get predictions for the simulator
X = ml_data.drop('cpu_cores', axis=1)
y = ml_data['cpu_cores']
lr = LinearRegression()
lr.fit(X, y)
ml_data['Forecast'] = lr.predict(X)

# Extract arrays for the simulator
actual_demand = ml_data['cpu_cores'].values
forecast_demand = ml_data['Forecast'].values


# 3. BUILD THE SIMULATOR
class CloudAutoscalingEnv(gym.Env):
    def __init__(self, actual_trace, forecast_trace):
        super(CloudAutoscalingEnv, self).__init__()

        self.actual_trace = actual_trace
        self.forecast_trace = forecast_trace
        self.max_steps = len(self.actual_trace)
        self.current_step = 0

        # System Constraints
        self.MAX_REPLICAS = 100
        self.MIN_REPLICAS = 1
        self.CAPACITY_PER_NODE = 64

        # Optimization Weights (Cost vs SLO)
        self.COST_WEIGHT = 1
        self.SLO_PENALTY_WEIGHT = 50 # High penalty for crashing

        # Actions: 0 = Scale Down (-1), 1 = Do Nothing (0), 2 = Scale Up (+1)
        self.action_space = spaces.Discrete(3)

        # State: [Current Load, Current Replicas, Forecast]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(3,), dtype=np.float32
        )
        self.current_replicas = 1

    def reset(self):
        self.current_step = 0
        self.current_replicas = 1
        return self._get_observation()

    def step(self, action):
        # 1. Apply Action (Map 0,1,2 to -1,0,+1)
        scaling_decision = action - 1
        self.current_replicas += scaling_decision

        # Enforce hardware limits
        self.current_replicas = np.clip(self.current_replicas, self.MIN_REPLICAS, self.MAX_REPLICAS)

        # 2. Calculate Metrics
        current_demand = self.actual_trace[self.current_step]
        total_capacity = self.current_replicas * self.CAPACITY_PER_NODE

        # 3. Calculate Reward
        cost_penalty = -1 * self.COST_WEIGHT * self.current_replicas

        slo_penalty = 0
        is_violation = False
        if current_demand > total_capacity:
            is_violation = True
            slo_penalty = -1 * self.SLO_PENALTY_WEIGHT * (current_demand - total_capacity)

        reward = cost_penalty + slo_penalty

        # 4. Advance Step
        self.current_step += 1
        done = self.current_step >= self.max_steps - 1

        info = {
            'demand': current_demand,
            'capacity': total_capacity,
            'replicas': self.current_replicas,
            'violation': is_violation,
            'reward': reward
        }

        return self._get_observation(), reward, done, info

    def _get_observation(self):
        demand = self.actual_trace[self.current_step]
        forecast = self.forecast_trace[self.current_step]
        return np.array([demand, self.current_replicas, forecast])

# 4. RUN THE HARNESS WITH A HEURISTIC AGENT
print("Initializing Simulator...")
env = CloudAutoscalingEnv(actual_demand, forecast_demand)

# A simple rule-based agent to prove the simulator works before adding RL
def heuristic_agent(obs):
    current_demand, replicas, forecast = obs
    capacity = replicas * env.CAPACITY_PER_NODE

    # Proactive Rule - If the forecast predicts we will crash, scale up
    if forecast > capacity:
        return 2 # Scale Up
    # Cost-saving Rule - If we have too much buffer, scale down
    elif capacity - forecast > (env.CAPACITY_PER_NODE * 2):
        return 0 # Scale Down
    else:
        return 1 # Do Nothing

obs = env.reset()
done = False
history = []

print("Running Simulation...")
while not done:
    action = heuristic_agent(obs)
    obs, reward, done, info = env.step(action)
    history.append(info)

results_df = pd.DataFrame(history)

# 5. VISUALIZE RESULTS
total_violations = results_df['violation'].sum()
print(f"Simulation Complete! Total SLO Violations: {total_violations}")

plt.figure(figsize=(14, 6))
# Plotting just a 500-step slice so the spikes are visible
plot_slice = results_df.iloc[1000:1500]

plt.plot(plot_slice.index, plot_slice['demand'], label='Actual CPU Demand', color='black', alpha=0.7)
plt.plot(plot_slice.index, plot_slice['capacity'], label='Allocated Capacity (Heuristic Policy)', color='blue', drawstyle='steps-post')
plt.fill_between(plot_slice.index, plot_slice['demand'], plot_slice['capacity'],
                 where=(plot_slice['demand'] > plot_slice['capacity']), color='red', alpha=0.3, label='SLO Violation (Crash)')

plt.title("Week 6 Testbed: Heuristic Agent + Linear Regression Forecast")
plt.xlabel("Time Steps (5-min intervals)")
plt.ylabel("CPU Cores")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('Week6_Simulator_Test.png')
plt.show()
