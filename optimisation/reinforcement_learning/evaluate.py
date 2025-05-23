import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from flight_scheduling_env import FlightSchedulingEnv
from prep_data_for_RL import prep_data
import matplotlib.pyplot as plt

def decode_action(flat_action, num_flights):
    # Each flight has: [w, aircraft, pilot1, pilot2, crew1, crew2, crew3]
    return np.array(flat_action).reshape(num_flights, 7)

def visualize_assignments(decoded_actions):
    num_flights = decoded_actions.shape[0]
    assigned_flights = np.where(decoded_actions[:, 0] == 1)[0]

    aircraft = decoded_actions[assigned_flights, 1]
    pilots = decoded_actions[assigned_flights, 2:4]
    crew = decoded_actions[assigned_flights, 4:7]

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axs[0].scatter(assigned_flights, aircraft, c='blue')
    axs[0].set_ylabel("Aircraft ID")
    axs[0].set_title("Aircraft Assignments")

    for i in range(2):
        axs[1].scatter(assigned_flights, pilots[:, i], label=f"Pilot {i+1}")
    axs[1].set_ylabel("Pilot ID")
    axs[1].legend()
    axs[1].set_title("Pilot Assignments")

    for i in range(3):
        axs[2].scatter(assigned_flights, crew[:, i], label=f"Crew {i+1}")
    axs[2].set_ylabel("Crew ID")
    axs[2].set_xlabel("Flight Index")
    axs[2].legend()
    axs[2].set_title("Crew Assignments")

    plt.tight_layout()
    plt.show()

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
model = PPO.load("ppo_flight_scheduler")

# Reset environment
obs = eval_env.reset()
total_rewards = []
all_infos = []

# Evaluation loop
for episode in range(10):
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

# Summary statistics
print("\n=== Evaluation Summary ===")
print(f"Average Reward: {np.mean(total_rewards):.2f}")
print(f"Min Reward: {np.min(total_rewards):.2f}")
print(f"Max Reward: {np.max(total_rewards):.2f}")