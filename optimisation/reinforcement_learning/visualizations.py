import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

def budget_visual(all_infos):
    budget_usages = [i["total_cost"] for i in all_infos]
    plt.hist(budget_usages, bins=10, color='skyblue', edgecolor='black')
    plt.xlabel("Total Cost")
    plt.ylabel("Frequency")
    plt.title("Distribution of Total Budget Usage per Episode")
    plt.grid(True)
    plt.show()

def emission_visual(all_infos, data):
    emissions = [i["emissions"] for i in all_infos]
    plt.plot(emissions, label='Emissions')
    plt.axhline(y=data["max_allowed_avg_emission"] * data["num_flights"], color='r', linestyle='--', label='Emission Limit')
    plt.xlabel("Episode")
    plt.ylabel("Total Emissions")
    plt.title("Emissions per Episode")
    plt.legend()
    plt.grid(True)
    plt.show()

def crew_overwork(all_infos):
    pilot_overworks = np.array([i["pilot_hours_over"] for i in all_infos])
    crew_overworks = np.array([i["crew_hours_over"] for i in all_infos])

    plt.figure(figsize=(12, 4))
    sns.heatmap(pilot_overworks, cmap="Reds", cbar_kws={"label": "Overwork Hours"})
    plt.xlabel("Pilot ID")
    plt.ylabel("Episode")
    plt.title("Pilot Overwork Heatmap")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    sns.heatmap(crew_overworks, cmap="Blues", cbar_kws={"label": "Overwork Hours"})
    plt.xlabel("Crew ID")
    plt.ylabel("Episode")
    plt.title("Crew Overwork Heatmap")
    plt.tight_layout()
    plt.show()


