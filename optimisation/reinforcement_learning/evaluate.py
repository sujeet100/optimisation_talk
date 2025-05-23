import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from flight_scheduling_env import FlightSchedulingEnv
from prep_data_for_RL import prep_data
import matplotlib.pyplot as plt

# Decode action from flat MultiDiscrete array
def decode_actions(action_flat, num_flights):
    action_array = np.array(action_flat).reshape(num_flights, 4)
    return action_array   # [ [w_i, aircraft, pilot, crew], ... ]

# Load data
csv_files = ["./data_sim_eval/aircraft_data.csv", "./data_sim_eval/pilot_data.csv", "./data_sim_eval/crew_data.csv", "./data_sim_eval/flight_data.csv"]
json_file = "./data_sim_eval/opt_params.json"
data = prep_data(csv_files, json_file)

# Load trained model
model = PPO.load("ppo_flight_scheduler")

# Create evaluation environment
eval_env = DummyVecEnv([lambda: FlightSchedulingEnv(data)])

# Reset environment
obs = eval_env.reset()

# Run evaluation episodes
n_eval_episodes = 10
total_rewards = []

for episode in range(n_eval_episodes):
    obs = eval_env.reset()
    done = False
    episode_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)

        decoded = decode_actions(action[0], data["num_flights"])
        assignments = []

        for i, (w, a_idx, p_idx, c_idx) in enumerate(decoded):
            if w == 1:
                assignments.append({
                    "Flight": i,
                    "Aircraft": a_idx,
                    "Pilot": p_idx,
                    "Crew": c_idx
                })
        df_assignments = pd.DataFrame(assignments)

        episode_reward += reward[0]
        done = done[0]

        # Optionally log detailed info
        # print(f"[Episode {episode + 1}] Step Reward: {reward[0]} | Info: {info[0]}")

    total_rewards.append(episode_reward)
    # print(f"[Episode {episode + 1}] Total Reward: {episode_reward}")

# Summary
print("\nEvaluation Summary")
print(f"Average Reward: {np.mean(total_rewards):.2f}")
print(f"Reward Std Dev : {np.std(total_rewards):.2f}")
