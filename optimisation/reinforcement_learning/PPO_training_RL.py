import os

import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import DummyVecEnv

from flight_scheduling_env import FlightSchedulingEnv
from prep_data_for_RL import prep_data

# Logging directory
log_dir = "./ppo_logs"
os.makedirs(log_dir, exist_ok=True)

# Initialize environment
csv_files = ["./data_sim/aircraft_data.csv", "./data_sim/pilot_data.csv", "./data_sim/crew_data.csv",
             "./data_sim/flight_data.csv"]
json_file = "./data_sim/opt_params.json"
data = prep_data(csv_files, json_file)
env = DummyVecEnv([lambda: Monitor(FlightSchedulingEnv(data), filename=log_dir)])

# Optional: check the custom environment
check_env(FlightSchedulingEnv(data), warn=True)

# Define and train PPO model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, learning_rate=1e-5, n_steps=2048, batch_size=64,
            ent_coef=0.01)
model.learn(total_timesteps=10000)

# Save model
model.save("ppo_flight_scheduler_visual")

# Evaluate model
eval_env = DummyVecEnv([lambda: FlightSchedulingEnv(data)])
obs = eval_env.reset()
for _ in range(10):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = eval_env.step(action)
    print(f"Reward: {rewards}, Info: {info}")

results = load_results(log_dir)
x, y = ts2xy(results, 'episodes')

plt.figure(figsize=(10, 5))
plt.plot(x, y)
plt.xlabel("Episodes")
plt.ylabel("Episode Reward")
plt.title("PPO Training Reward Over Time")
plt.grid()
plt.show()

print("something")
