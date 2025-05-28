import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from flight_scheduling_env2 import FlightSchedulingEnv
from prep_data_for_RL import prep_data
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def visualize_episode_with_overuse(env, episode_log):
    num_flights = env.envs[0].num_flights
    budget_cap = env.envs[0].budget_cap
    emission_cap = env.envs[0].emission_limit * num_flights
    Hp = env.envs[0].data["max_hours_pilot"]
    Hc = env.envs[0].data["max_hours_crew"]

    flight_ids = []
    aircraft_assignments = []
    pilot1_assignments = []
    pilot2_assignments = []
    crew1_assignments = []
    crew2_assignments = []
    crew3_assignments = []
    emission_progress = []
    cost_progress = []

    total_emission = 0
    total_cost = 0
    pilot_hours = np.zeros(env.envs[0].num_pilots)
    crew_hours = np.zeros(env.envs[0].num_crew)

    for flight_id, action, reward, info in episode_log:
        flight_ids.append(flight_id)
        w_i, a_idx, p1_idx, p2_idx, c1_idx, c2_idx, c3_idx = action[0][0]

        if w_i == 1:
            aircraft_assignments.append(a_idx)
            pilot1_assignments.append(p1_idx)
            pilot2_assignments.append(p2_idx)
            crew1_assignments.append(c1_idx)
            crew2_assignments.append(c2_idx)
            crew3_assignments.append(c3_idx)

            # Update hours
            Di = env.envs[0].data["flight_duration"][flight_id]
            pilot_hours[p1_idx] += Di
            pilot_hours[p2_idx] += Di
            crew_hours[c1_idx] += Di
            crew_hours[c2_idx] += Di
            crew_hours[c3_idx] += Di
        else:
            aircraft_assignments.append(-1)
            pilot1_assignments.append(-1)
            pilot2_assignments.append(-1)
            crew1_assignments.append(-1)
            crew2_assignments.append(-1)
            crew3_assignments.append(-1)

        total_emission += info[0].get("emissions", 0)
        total_cost += info[0].get("total_cost", 0)

        emission_progress.append(min(total_emission, emission_cap))
        cost_progress.append(min(total_cost, budget_cap))

    fig, axs = plt.subplots(4, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [1, 1.2, 1.2, 1.5]}, sharex=True)

    axs[0].plot(flight_ids, aircraft_assignments, 'o-', label='Aircraft ID')
    axs[0].set_ylabel("Aircraft")
    axs[0].legend()

    axs[1].plot(flight_ids, pilot1_assignments, 'x-', label='Pilot 1')
    axs[1].plot(flight_ids, pilot2_assignments, 'x-', label='Pilot 2')
    axs[1].plot(flight_ids, crew1_assignments, '+-', label='Crew 1')
    axs[1].plot(flight_ids, crew2_assignments, '+-', label='Crew 2')
    axs[1].plot(flight_ids, crew3_assignments, '+-', label='Crew 3')
    axs[1].set_ylabel("Assignments")
    axs[1].legend()

    axs[2].plot(flight_ids, cost_progress, label='Cost Accumulated')
    axs[2].plot(flight_ids, emission_progress, label='Emissions Accumulated')
    axs[2].hlines(budget_cap, 0, num_flights - 1, colors='red', linestyles='dashed', label='Budget Cap')
    axs[2].hlines(emission_cap, 0, num_flights - 1, colors='green', linestyles='dashed', label='Emission Cap')
    axs[2].set_ylabel("Cumulative")
    axs[2].legend()

    # Plot overused pilots and crew
    width = 0.35
    x1 = np.arange(env.envs[0].num_pilots)
    x2 = np.arange(env.envs[0].num_crew)

    axs[3].bar(x1, pilot_hours, width=width, label="Pilot Hours", color=["red" if h > Hp else "skyblue" for h in pilot_hours])
    axs[3].bar(x2 + env.envs[0].num_pilots + 1, crew_hours, width=width, label="Crew Hours", color=["red" if h > Hc else "lightgreen" for h in crew_hours])
    axs[3].hlines(Hp, xmin=0, xmax=env.envs[0].num_pilots - 1, colors='gray', linestyles='dashed')
    axs[3].hlines(Hc, xmin=env.envs[0].num_pilots + 1, xmax=env.envs[0].num_pilots + env.envs[0].num_crew, colors='gray', linestyles='dashed')
    axs[3].set_ylabel("Logged Hours")
    axs[3].set_xlabel("Pilot and Crew IDs")
    axs[3].legend()
    axs[3].set_xticks(
        list(x1) + list(x2 + env.envs[0].num_pilots + 1),
        labels=[f"P{i}" for i in x1] + [f"C{i}" for i in x2],
        rotation=90
    )

    plt.suptitle("Flight Assignment, Resource Usage, and Overuse Visualization")
    plt.tight_layout()
    plt.show()


# Main evaluation code
def main():
    # Load data and environment
    csv_files = [
        "./data_sim_eval/aircraft_data.csv",
        "./data_sim_eval/pilot_data.csv",
        "./data_sim_eval/crew_data.csv",
        "./data_sim_eval/flight_data.csv"
    ]
    json_file = "./data_sim_eval/opt_params.json"

    try:
        data = prep_data(csv_files, json_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Create environment
    try:
        eval_env = DummyVecEnv([lambda: FlightSchedulingEnv(data)])
        check_env(FlightSchedulingEnv(data), warn=True)
        num_flights = data.get("num_flights", 0)
        print(f"Number of flights: {num_flights}")
    except Exception as e:
        print(f"Error creating environment: {e}")
        return

    # Load trained model
    try:
        model = PPO.load("ppo_flight_scheduler")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize tracking variables
    episode_log = []
    obs = eval_env.reset()
    done = False

    while not done:
        action = model.predict(obs)  # your trained policy
        obs, reward, done, info = eval_env.step(action)
        episode_log.append((eval_env.envs[0].current_flight - 1, action, reward, info))

    print(episode_log)
    visualize_episode_with_overuse(eval_env, episode_log)


if __name__ == "__main__":
    main()