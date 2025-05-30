import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle
import numpy as np

def visualize_solution(best_individual, problem, fitness_history):
    """Create comprehensive visualizations of the optimized solution"""

    # Decode the best solution
    genome = best_individual.genome
    n_flights = problem.n_flights

    # Extract assignments
    flight_data = []
    for i in range(n_flights):
        base_idx = i * 6
        flight_info = {
            'flight_id': i + 1,
            'aircraft': genome[base_idx],
            'pilot1': genome[base_idx + 1],
            'pilot2': genome[base_idx + 2],
            'crew1': genome[base_idx + 3],
            'crew2': genome[base_idx + 4],
            'crew3': genome[base_idx + 5],
            'revenue': problem.flights['revenue'][i],
            'duration': problem.flights['duration'][i],
            'aircraft_cost': problem.aircraft['cost'][genome[base_idx]],
            'emission_rate': problem.aircraft['emission_rate'][genome[base_idx]]
        }
        flight_data.append(flight_info)

    # Flight assignments
    flight_ids = np.array([info['flight_id'] for info in flight_data])
    aircraft_ids = np.array([info['aircraft'] for info in flight_data])
    pilot1_ids = np.array([info['pilot1'] for info in flight_data])
    pilot2_ids = np.array([info['pilot2'] for info in flight_data])
    crew1_ids = np.array([info['crew1'] for info in flight_data])
    crew2_ids = np.array([info['crew2'] for info in flight_data])
    crew3_ids = np.array([info['crew3'] for info in flight_data])
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    fig.suptitle('Flight Resource Assignments', fontsize=16, fontweight='bold')
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    ax1 = axes[0]
    scatter1 = ax1.scatter(flight_ids, aircraft_ids,
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
    ax1.set_yticks(np.arange(int(min(aircraft_ids)), int(max(aircraft_ids)) + 1))

    # Add text annotations for aircraft assignments
    for i, (x, y) in enumerate(zip(flight_ids, aircraft_ids)):
        if i % max(1, len(flight_ids) // 20) == 0:  # Annotate every nth point to avoid crowding
            ax1.annotate(f'F{x}→A{y}', (x, y), xytext=(5, 5),
                         textcoords='offset points', fontsize=8, alpha=0.7)

    # Subplot 2: Pilot Assignments
    ax2 = axes[1]
    scatter2a = ax2.scatter(flight_ids, pilot1_ids,
                            c=colors[1], alpha=0.8, s=80, label='Pilot 1',
                            edgecolors='black', linewidth=0.5, marker='o')
    scatter2b = ax2.scatter(flight_ids, pilot2_ids,
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
    all_pilot_ids = np.concatenate([pilot1_ids, pilot2_ids])
    ax2.set_yticks(np.arange(int(min(all_pilot_ids)), int(max(all_pilot_ids)) + 1))

    # Subplot 3: Crew Assignments
    ax3 = axes[2]
    scatter3a = ax3.scatter(flight_ids, crew1_ids,
                            c=colors[2], alpha=0.8, s=80, label='Crew 1',
                            edgecolors='black', linewidth=0.5, marker='o')
    scatter3b = ax3.scatter(flight_ids, crew2_ids,
                            c='#8A2BE2', alpha=0.8, s=80, label='Crew 2',
                            edgecolors='black', linewidth=0.5, marker='^')
    scatter3c = ax3.scatter(flight_ids, crew3_ids,
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
    all_crew_ids = np.concatenate([crew1_ids, crew2_ids, crew3_ids])
    ax3.set_yticks(np.arange(int(min(all_crew_ids)), int(max(all_crew_ids)) + 1))

    plt.tight_layout()
    plt.show()


    #Resource Utilization Heatmap
    # 1. Flight assignments
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Flight Resource Assignments', fontsize=16, fontweight='bold')
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    ax1 = axes[0]
    scatter1 = ax1.scatter(flight_ids, aircraft_ids,
                           c=colors[0], alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Flight ID', fontsize=12)
    ax1.set_ylabel('Aircraft ID', fontsize=12)
    ax1.set_title('Aircraft Assignments', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Create resource utilization matrix
    resource_matrix = np.zeros((2, max(problem.n_pilots, problem.n_crew)))

    # Pilot utilization
    pilot_hours = np.zeros(problem.n_pilots)
    for flight in flight_data:
        pilot_hours[flight['pilot1']] += flight['duration']
        pilot_hours[flight['pilot2']] += flight['duration']
    resource_matrix[0, :len(pilot_hours)] = pilot_hours

    # Crew utilization
    crew_hours = np.zeros(problem.n_crew)
    for flight in flight_data:
        crew_hours[flight['crew1']] += flight['duration']
        crew_hours[flight['crew2']] += flight['duration']
        crew_hours[flight['crew3']] += flight['duration']
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

