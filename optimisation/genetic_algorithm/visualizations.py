
def visualize_solution(best_individual, problem, fitness_history):
    """Create comprehensive visualizations of the optimized solution"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from matplotlib.patches import Rectangle
    import numpy as np

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

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

    # Create subplots
    fig = plt.figure(figsize=(20, 16))

    # 1. Gantt Chart - Flight Schedule Timeline
    ax1 = plt.subplot(3, 4, (1, 2))

    # Simulate flight start times (you can replace with actual times)
    start_times = np.cumsum([0] + [flight_data[i]['duration'] for i in range(n_flights - 1)])

    colors = plt.cm.Set3(np.linspace(0, 1, problem.n_aircraft))

    for i, flight in enumerate(flight_data):
        aircraft_id = flight['aircraft']
        color = colors[aircraft_id]

        # Draw flight bar
        rect = Rectangle((start_times[i], i), flight['duration'], 0.8,
                         facecolor=color, edgecolor='black', alpha=0.7)
        ax1.add_patch(rect)

        # Add flight label
        ax1.text(start_times[i] + flight['duration'] / 2, i + 0.4,
                 f"F{flight['flight_id']}\nA{aircraft_id}",
                 ha='center', va='center', fontsize=8, fontweight='bold')

    ax1.set_xlim(0, max(start_times) + max(f['duration'] for f in flight_data))
    ax1.set_ylim(-0.5, n_flights - 0.5)
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Flights')
    ax1.set_title('Flight Schedule Timeline (Gantt Chart)')
    ax1.grid(True, alpha=0.3)

    # Add legend for aircraft
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], label=f'Aircraft {i}')
                       for i in range(problem.n_aircraft)]
    ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

    # 2. Resource Utilization Heatmap
    ax2 = plt.subplot(3, 4, (3, 4))

    # Create resource utilization matrix
    resource_matrix = np.zeros((3, max(problem.n_pilots, problem.n_crew, problem.n_aircraft)))

    # Aircraft utilization
    for flight in flight_data:
        resource_matrix[0, flight['aircraft']] += flight['duration']

    # Pilot utilization
    pilot_hours = np.zeros(problem.n_pilots)
    for flight in flight_data:
        pilot_hours[flight['pilot1']] += flight['duration']
        pilot_hours[flight['pilot2']] += flight['duration']
    resource_matrix[1, :len(pilot_hours)] = pilot_hours

    # Crew utilization
    crew_hours = np.zeros(problem.n_crew)
    for flight in flight_data:
        crew_hours[flight['crew1']] += flight['duration']
        crew_hours[flight['crew2']] += flight['duration']
        crew_hours[flight['crew3']] += flight['duration']
    resource_matrix[2, :len(crew_hours)] = crew_hours

    im = ax2.imshow(resource_matrix, cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(range(resource_matrix.shape[1]))
    ax2.set_xticklabels([f'R{i}' for i in range(resource_matrix.shape[1])])
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Aircraft', 'Pilots', 'Crew'])
    ax2.set_title('Resource Utilization Heatmap (Hours)')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Hours Assigned')

    # Add text annotations
    for i in range(resource_matrix.shape[0]):
        for j in range(resource_matrix.shape[1]):
            if resource_matrix[i, j] > 0:
                ax2.text(j, i, f'{resource_matrix[i, j]:.1f}',
                         ha='center', va='center', color='white', fontweight='bold')

    # 3. Assignment Matrix Visualization
    ax3 = plt.subplot(3, 4, 5)

    # Create flight-resource assignment matrix
    assignment_data = []
    for flight in flight_data:
        assignment_data.append([
            flight['flight_id'],
            flight['aircraft'],
            flight['pilot1'],
            flight['pilot2'],
            flight['crew1'],
            flight['crew2'],
            flight['crew3']
        ])

    df_assignments = pd.DataFrame(assignment_data,
                                  columns=['Flight', 'Aircraft', 'Pilot1', 'Pilot2', 'Crew1', 'Crew2', 'Crew3'])

    # Create a heatmap-style visualization
    assignment_matrix = df_assignments.iloc[:, 1:].values
    im3 = ax3.imshow(assignment_matrix.T, cmap='tab20', aspect='auto')

    ax3.set_xticks(range(len(flight_data)))
    ax3.set_xticklabels([f'F{i + 1}' for i in range(len(flight_data))])
    ax3.set_yticks(range(6))
    ax3.set_yticklabels(['Aircraft', 'Pilot1', 'Pilot2', 'Crew1', 'Crew2', 'Crew3'])
    ax3.set_title('Flight-Resource Assignment Matrix')

    # Add assignment numbers
    for i in range(assignment_matrix.shape[1]):
        for j in range(assignment_matrix.shape[0]):
            ax3.text(i, j, str(assignment_matrix[j, i]),
                     ha='center', va='center', fontweight='bold', fontsize=10)

    # 4. Revenue vs Cost Analysis
    ax4 = plt.subplot(3, 4, 6)

    revenues = [f['revenue'] for f in flight_data]
    costs = []

    for flight in flight_data:
        flight_cost = (flight['aircraft_cost'] +
                       problem.pilots['salary_per_hour'][flight['pilot1']] * flight['duration'] +
                       problem.pilots['salary_per_hour'][flight['pilot2']] * flight['duration'] +
                       problem.crew['salary_per_hour'][flight['crew1']] * flight['duration'] +
                       problem.crew['salary_per_hour'][flight['crew2']] * flight['duration'] +
                       problem.crew['salary_per_hour'][flight['crew3']] * flight['duration'])
        costs.append(flight_cost)

    x = np.arange(len(flight_data))
    width = 0.35

    bars1 = ax4.bar(x - width / 2, revenues, width, label='Revenue', color='green', alpha=0.7)
    bars2 = ax4.bar(x + width / 2, costs, width, label='Cost', color='red', alpha=0.7)

    ax4.set_xlabel('Flights')
    ax4.set_ylabel('Amount ($)')
    ax4.set_title('Revenue vs Cost by Flight')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'F{i + 1}' for i in range(len(flight_data))])
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                 f'${height:,.0f}', ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                 f'${height:,.0f}', ha='center', va='bottom', fontsize=8)

    # 5. Emissions Analysis
    ax5 = plt.subplot(3, 4, 7)

    emissions = [f['emission_rate'] * f['duration'] for f in flight_data]
    avg_emission = sum(emissions) / len(emissions)

    bars = ax5.bar(range(len(flight_data)), emissions, color='orange', alpha=0.7)
    ax5.axhline(y=avg_emission, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_emission:.1f}')
    ax5.axhline(y=problem.constraints['avg_emission_max'], color='purple', linestyle='--',
                linewidth=2, label=f'Max Allowed: {problem.constraints["avg_emission_max"]}')

    ax5.set_xlabel('Flights')
    ax5.set_ylabel('Total Emissions')
    ax5.set_title('Emission Analysis by Flight')
    ax5.set_xticks(range(len(flight_data)))
    ax5.set_xticklabels([f'F{i + 1}' for i in range(len(flight_data))])
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Workload Balance Analysis
    ax6 = plt.subplot(3, 4, 8)

    # Calculate total workload including logged hours
    pilot_workload = np.array(problem.pilots['logged_hours']) + pilot_hours
    crew_workload = np.array(problem.crew['logged_hours']) + crew_hours[:len(problem.crew['logged_hours'])]

    # Create workload comparison
    x_pilots = np.arange(len(pilot_workload))
    x_crew = np.arange(len(crew_workload)) + len(pilot_workload) + 1

    bars1 = ax6.bar(x_pilots, pilot_workload, color='blue', alpha=0.7, label='Pilots')
    bars2 = ax6.bar(x_crew, crew_workload, color='green', alpha=0.7, label='Crew')

    # Add max hours line
    ax6.axhline(y=problem.constraints['max_hours_pilot'], color='red', linestyle='--',
                alpha=0.7, label=f'Max Pilot Hours: {problem.constraints["max_hours_pilot"]}')
    ax6.axhline(y=problem.constraints['max_hours_crew'], color='orange', linestyle='--',
                alpha=0.7, label=f'Max Crew Hours: {problem.constraints["max_hours_crew"]}')

    all_x = list(x_pilots) + list(x_crew)
    all_labels = [f'P{i}' for i in range(len(pilot_workload))] + [f'C{i}' for i in range(len(crew_workload))]

    ax6.set_xticks(all_x)
    ax6.set_xticklabels(all_labels, rotation=45)
    ax6.set_ylabel('Total Hours')
    ax6.set_title('Personnel Workload Analysis')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. Fitness Evolution
    ax7 = plt.subplot(3, 4, 9)
    ax7.plot(fitness_history, linewidth=2, color='purple')
    ax7.set_xlabel('Generation')
    ax7.set_ylabel('Fitness')
    ax7.set_title('Fitness Evolution')
    ax7.grid(True, alpha=0.3)

    # 8. Solution Summary Table
    ax8 = plt.subplot(3, 4, (10, 11))
    ax8.axis('off')

    # Calculate summary statistics
    total_revenue = sum(revenues)
    total_cost = sum(costs)
    net_profit = total_revenue - total_cost
    total_emissions = sum(emissions)
    avg_emissions = total_emissions / len(emissions)

    constraint_violations = 0
    if total_cost > problem.constraints['budget']:
        constraint_violations += 1
    if avg_emissions > problem.constraints['avg_emission_max']:
        constraint_violations += 1
    if any(h > problem.constraints['max_hours_pilot'] for h in pilot_workload):
        constraint_violations += 1
    if any(h > problem.constraints['max_hours_crew'] for h in crew_workload):
        constraint_violations += 1

    summary_text = f"""
    OPTIMIZATION SUMMARY
    ==================

    Financial Performance:
    • Total Revenue: ${total_revenue:,.2f}
    • Total Cost: ${total_cost:,.2f}
    • Net Profit: ${net_profit:,.2f}
    • Budget Utilization: {(total_cost / problem.constraints['budget'] * 100):.1f}%

    Environmental Impact:
    • Total Emissions: {total_emissions:.1f}
    • Average Emissions: {avg_emissions:.1f}
    • Emission Limit: {problem.constraints['avg_emission_max']}

    Resource Utilization:
    • Flights Scheduled: {len(flight_data)}
    • Aircraft Used: {len(set(f['aircraft'] for f in flight_data))} / {problem.n_aircraft}
    • Pilots Used: {len(set([f['pilot1'] for f in flight_data] + [f['pilot2'] for f in flight_data]))} / {problem.n_pilots}
    • Crew Used: {len(set([f['crew1'] for f in flight_data] + [f['crew2'] for f in flight_data] + [f['crew3'] for f in flight_data]))} / {problem.n_crew}

    Constraint Status:
    • Violations: {constraint_violations}
    • Final Fitness: {best_individual.fitness:.2f}
    """

    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    # 9. Network Graph (Flight-Resource Connections)
    ax9 = plt.subplot(3, 4, 12)

    # Simple network visualization showing connections
    # This is a simplified version - you could use networkx for more sophisticated graphs

    # Position nodes
    flight_positions = [(i, 2) for i in range(len(flight_data))]
    aircraft_positions = [(i * len(flight_data) / problem.n_aircraft, 3) for i in range(problem.n_aircraft)]
    pilot_positions = [(i * len(flight_data) / problem.n_pilots, 1) for i in range(problem.n_pilots)]
    crew_positions = [(i * len(flight_data) / problem.n_crew, 0) for i in range(problem.n_crew)]

    # Draw nodes
    for i, pos in enumerate(flight_positions):
        ax9.scatter(pos[0], pos[1], s=100, c='red', marker='s', label='Flights' if i == 0 else '')
        ax9.text(pos[0], pos[1], f'F{i + 1}', ha='center', va='center', fontsize=8, fontweight='bold')

    for i, pos in enumerate(aircraft_positions):
        ax9.scatter(pos[0], pos[1], s=80, c='blue', marker='^', label='Aircraft' if i == 0 else '')
        ax9.text(pos[0], pos[1] + 0.1, f'A{i}', ha='center', va='bottom', fontsize=8)

    # Draw connections (simplified)
    for i, flight in enumerate(flight_data):
        flight_pos = flight_positions[i]
        aircraft_pos = aircraft_positions[flight['aircraft']]
        ax9.plot([flight_pos[0], aircraft_pos[0]], [flight_pos[1], aircraft_pos[1]],
                 'b-', alpha=0.3, linewidth=1)

    ax9.set_xlim(-0.5, len(flight_data) - 0.5)
    ax9.set_ylim(-0.5, 3.5)
    ax9.set_title('Resource Assignment Network')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig