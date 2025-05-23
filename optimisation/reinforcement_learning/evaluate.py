import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from flight_scheduling_env import FlightSchedulingEnv  # import your environment

# Load the trained model
model = PPO.load("ppo_flight_scheduler")

# Load or define the same environment data used during training
data = {
    "num_flights": 100,
    "num_aircraft": 5,
    "num_pilots": 10,
    "num_crew": 10,
    "budget_cap": 50000,
    "max_allowed_avg_emission": 1000,
    "flight_revenue": np.random.randint(1000, 5000, size=100),
    "flight_priority": np.random.rand(100),
    "flight_duration": np.random.randint(1, 5, size=100),
    "op_cost_per_km": np.random.rand(5) * 10,
    "carbon_emission_gm_per_km": np.random.rand(5) * 100,
    "fuel_efficiency_km_per_kg": np.random.rand(5) * 2,
    "salary_pilot": np.random.rand(10) * 50,
    "logged_hours_pilot": np.zeros(10),
    "salary_crew": np.random.rand(10) * 30,
    "logged_hours_crew": np.zeros(10),
    "weather_based_fuel_degradation_factor": np.random.rand(100) + 1,
    "weather_based_emission_amplification_factor": np.random.rand(100) + 1,
    "max_hours_pilot": np.ones(10) * 8,
    "max_hours_crew": np.ones(10) * 8,
}

# Create env instance
env = FlightSchedulingEnv(data)
obs, _ = env.reset()

# Evaluate
obs_list, rewards, emissions, costs, budget_violations, emission_violations = [], [], [], [], [], []

obs = obs.astype(np.float32)
action, _ = model.predict(obs, deterministic=True)
obs, reward, terminated, truncated, info = env.step(action)

# Store results
obs_list.append(obs)
rewards.append(reward)
emissions.append(info["emissions"])
costs.append(info["total_cost"])
budget_violations.append(info["budget_overrun"])
emission_violations.append(info["emission_violation"])

# Visualize
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.title("Reward")
plt.bar(["Reward"], rewards)

plt.subplot(2, 2, 2)
plt.title("Emissions")
plt.bar(["Emissions"], emissions)

plt.subplot(2, 2, 3)
plt.title("Total Cost")
plt.bar(["Cost"], costs)

plt.subplot(2, 2, 4)
plt.title("Constraint Violations")
plt.bar(["Budget Overrun", "Emission Violation"], [budget_violations[0], emission_violations[0]])

plt.tight_layout()
plt.show()
