import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from flight_scheduling_env import FlightSchedulingEnv
from prep_data_for_RL import prep_data
import matplotlib.pyplot as plt

from visualizations import decode_action, visualize_assignments, crew_overwork, budget_visual, emission_visual

# Load data and environment
csv_files = [
    "./data_sim_eval/aircraft_data.csv",
    "./data_sim_eval/pilot_data.csv",
    "./data_sim_eval/crew_data.csv",
    "./data_sim_eval/flight_data.csv"
]
json_file = "./data_sim_eval/opt_params.json"
data = prep_data(csv_files, json_file)

# Create environment
eval_env = DummyVecEnv([lambda: FlightSchedulingEnv(data)])
num_flights = data["num_flights"]

# Load trained model
model = PPO.load("ppo_flight_scheduler_visual")

# Reset environment
obs = eval_env.reset()
total_rewards = []
all_infos = []

# Evaluation loop
for episode in range(100):
    obs = eval_env.reset()
    done = False
    ep_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)

        raw_action = action[0]  # unwrap from DummyVecEnv
        reshaped_action = np.array(raw_action).reshape(num_flights, 7)

        obs, reward, done, info = eval_env.step([raw_action])
        done = done[0]

        ep_reward += reward[0]
        all_infos.append(info[0])

    total_rewards.append(ep_reward)

    print(f"\nEpisode {episode + 1} Reward: {ep_reward}")
    print("Assignments for first 5 flights:")
    for i, (w, a, p1, p2, c1, c2, c3) in enumerate(reshaped_action[:5]):
        print(
            f"Flight {i}: Scheduled={bool(w)}, Aircraft={a}, "
            f"Pilots=({p1}, {p2}), Crew=({c1}, {c2}, {c3})"
        )
    print("Info:", info[0])

decoded_actions = decode_action(raw_action, num_flights=50)
visualize_assignments(decoded_actions)
crew_overwork(all_infos)
budget_visual(all_infos)
emission_visual(all_infos, data)


# Summary statistics
print("\n=== Evaluation Summary ===")
print(f"Average Reward: {np.mean(total_rewards):.2f}")
print(f"Min Reward: {np.min(total_rewards):.2f}")
print(f"Max Reward: {np.max(total_rewards):.2f}")

plt.figure(figsize=(10, 4))
plt.plot(total_rewards, marker='o')
plt.xlabel("Episode")
plt.ylabel("Total Episode Reward")
plt.title("Evaluation Episode Rewards")
plt.grid(True)
plt.tight_layout()
plt.show()