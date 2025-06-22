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

from reinforcement_learning.data_generator import num_pilots, num_crew

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

    # Flight assignments
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    fig.suptitle('Flight Resource Assignments', fontsize=16, fontweight='bold')
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    ax1 = axes[0]
    scatter1 = ax1.scatter(flight_ids, aircraft_assignments,
                           c=colors[0], alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Flight ID', fontsize=12)
    ax1.set_ylabel('Aircraft ID', fontsize=12)
    ax1.set_title('Aircraft Assignments', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    # Set integer ticks for better readability
    if len(flight_ids) > 10:
        ax1.set_xticks(np.arange(int(min(flight_ids)), int(max(flight_ids)) + 1,
                                 max(1, len(flight_ids) // 10)))
    else:
        ax1.set_xticks(flight_ids)
    ax1.set_yticks(np.arange(int(min(aircraft_assignments)), int(max(aircraft_assignments)) + 1))
    # Add text annotations for aircraft assignments
    for i, (x, y) in enumerate(zip(flight_ids, aircraft_assignments)):
        if i % max(1, len(flight_ids) // 20) == 0:  # Annotate every nth point to avoid crowding
            ax1.annotate(f'F{x}→A{y}', (x, y), xytext=(5, 5),
                         textcoords='offset points', fontsize=8, alpha=0.7)

    # Subplot 2: Pilot Assignments
    ax2 = axes[1]
    scatter2a = ax2.scatter(flight_ids, pilot1_assignments,
                            c=colors[1], alpha=0.8, s=80, label='Pilot 1',
                            edgecolors='black', linewidth=0.5, marker='o')
    scatter2b = ax2.scatter(flight_ids, pilot2_assignments,
                            c='#2E8B57', alpha=0.8, s=80, label='Pilot 2',
                            edgecolors='black', linewidth=0.5, marker='^')
    ax2.set_xlabel('Flight ID', fontsize=12)
    ax2.set_ylabel('Pilot ID', fontsize=12)
    ax2.set_title('Pilot Assignments', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    # Set integer ticks
    if len(flight_ids) > 10:
        ax2.set_xticks(np.arange(int(min(flight_ids)), int(max(flight_ids)) + 1,
                                 max(1, len(flight_ids) // 10)))
    else:
        ax2.set_xticks(flight_ids)
    all_pilot_ids = np.concatenate([pilot1_assignments, pilot2_assignments])
    ax2.set_yticks(np.arange(int(min(all_pilot_ids)), int(max(all_pilot_ids)) + 1))

    # Subplot 3: Crew Assignments
    ax3 = axes[2]
    scatter3a = ax3.scatter(flight_ids, crew1_assignments,
                            c=colors[2], alpha=0.8, s=80, label='Crew 1',
                            edgecolors='black', linewidth=0.5, marker='o')
    scatter3b = ax3.scatter(flight_ids, crew2_assignments,
                            c='#8A2BE2', alpha=0.8, s=80, label='Crew 2',
                            edgecolors='black', linewidth=0.5, marker='^')
    scatter3c = ax3.scatter(flight_ids, crew3_assignments,
                            c='#FF1493', alpha=0.8, s=80, label='Crew 3',
                            edgecolors='black', linewidth=0.5, marker='s')

    ax3.set_xlabel('Flight ID', fontsize=12)
    ax3.set_ylabel('Crew ID', fontsize=12)
    ax3.set_title('Crew Assignments', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    # Set integer ticks
    if len(flight_ids) > 10:
        ax3.set_xticks(np.arange(int(min(flight_ids)), int(max(flight_ids)) + 1,
                                 max(1, len(flight_ids) // 10)))
    else:
        ax3.set_xticks(flight_ids)
    all_crew_ids = np.concatenate([crew1_assignments, crew2_assignments, crew3_assignments])
    ax3.set_yticks(np.arange(int(min(all_crew_ids)), int(max(all_crew_ids)) + 1))

    plt.tight_layout()
    plt.show()

    #Resource Utilization Heatmap
    # Create resource utilization matrix
    resource_matrix = np.zeros((2, max(num_pilots, num_crew)))
    # Pilot utilization
    resource_matrix[0, :len(pilot_hours)] = pilot_hours
    # Crew utilization
    resource_matrix[1, :len(crew_hours)] = crew_hours
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(20, 4))

    # Create the heatmap
    im = ax.imshow(resource_matrix, cmap='YlOrRd', aspect='auto')

    # Set x-axis ticks and labels
    ax.set_xticks(np.arange(resource_matrix.shape[1]))
    ax.set_xticklabels([f'R{i}' for i in range(resource_matrix.shape[1])])

    # Set y-axis ticks and labels
    ax.set_yticks(np.arange(resource_matrix.shape[0]))
    ax.set_yticklabels(['Pilots', 'Crew'])

    # Set title
    ax.set_title('Resource Utilization Heatmap (Hours)', fontsize=14, pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Hours Assigned', fontsize=12)

    # Ensure proper spacing and grid alignment
    ax.set_xlim(-0.5, resource_matrix.shape[1] - 0.5)
    ax.set_ylim(-0.5, resource_matrix.shape[0] - 0.5)

    # Optional: Add grid lines for better visualization
    ax.set_xticks(np.arange(resource_matrix.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(resource_matrix.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    # Adjust layout and display
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

    print(f'Summary: Total reward: {episode_log[-1][2]}; Total cost of operation: {episode_log[-1][3][0]['total_cost']}; '
          f'Average carbon emission: {episode_log[-1][3][0]['emissions']}')
    visualize_episode_with_overuse(eval_env, episode_log)


if __name__ == "__main__":
    main()