import gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from flight_scheduling_env import FlightSchedulingEnv
from prep_data_for_RL import prep_data

# Initialize environment
data = prep_data()
env = DummyVecEnv([lambda: FlightSchedulingEnv(data)])

# Optional: check the custom environment
check_env(FlightSchedulingEnv(data), warn=True)

# Define and train PPO model
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64, ent_coef=0.01)
model.learn(total_timesteps=100000)

# Save model
model.save("ppo_flight_scheduler")

# Evaluate model
eval_env = DummyVecEnv([lambda: FlightSchedulingEnv(data)])
obs = eval_env.reset()
for _ in range(10):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = eval_env.step(action)
    print(f"Reward: {rewards}, Info: {info}")
