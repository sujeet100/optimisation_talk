import random

import numpy as np
import matplotlib.pyplot as plt
from leap_ec import Individual
from leap_ec.decoder import IdentityDecoder
from prettytable import PrettyTable

from visualizations import visualize_solution

from data_generator import simulate_data


class FlightSchedulingProblem:
    def __init__(self, flights_data, aircraft_data, pilot_data, crew_data, constraints):
        """
        Initialize the flight scheduling optimization problem

        Parameters:
        - flights_data: dict with 'revenue' (Ri) and 'duration' (Di) lists
        - aircraft_data: dict with 'cost' (Cj) and 'emission_rate' (Ej) lists
        - pilot_data: dict with 'salary_per_hour' (Sm_p) and 'logged_hours' (Hm_p) lists
        - crew_data: dict with 'salary_per_hour' (Sn_c) and 'logged_hours' (Hn_c) lists
        - constraints: dict with budget (B), max_hours_pilot (Hp), max_hours_crew (Hc),
                      avg_emission_max (Eavg_max), penalties (lambda_budget, lambda_work)
        """
        self.flights = flights_data
        self.aircraft = aircraft_data
        self.pilots = pilot_data
        self.crew = crew_data
        self.constraints = constraints

        self.n_flights = len(flights_data['revenue'])
        self.n_aircraft = len(aircraft_data['cost'])
        self.n_pilots = len(pilot_data['salary_per_hour'])
        self.n_crew = len(crew_data['salary_per_hour'])

        # Genome structure: [aircraft_assignments, pilot1_assignments, pilot2_assignments, crew1_assignments, crew2_assignments, crew3_assignments]
        # Each assignment is a list of indices for each flight
        self.genome_length = self.n_flights * 6  # 1 aircraft + 2 pilots + 3 crew per flight
        self.genome_bound = []

    def create_individual(self):
        """Create a random valid individual"""
        genome = []

        # For each flight, assign aircraft and personnel
        for flight_idx in range(self.n_flights):
            # Aircraft assignment (one per flight)
            aircraft_choice = random.randint(0, self.n_aircraft - 1)
            genome.append(aircraft_choice)

            # Pilot assignments (2 different pilots per flight)
            pilot_choices = random.sample(range(self.n_pilots), min(2, self.n_pilots))
            while len(pilot_choices) < 2:  # Handle case where n_pilots < 2
                pilot_choices.append(random.randint(0, self.n_pilots - 1))
            genome.extend(pilot_choices)

            # Crew assignments (3 different crew members per flight)
            crew_choices = random.sample(range(self.n_crew), min(3, self.n_crew))
            while len(crew_choices) < 3:  # Handle case where n_crew < 3
                crew_choices.append(random.randint(0, self.n_crew - 1))
            genome.extend(crew_choices)

        return genome

    def evaluate_fitness(self, individual):
        """Evaluate the fitness of an individual solution"""
        genome = individual.genome

        # Decode genome
        aircraft_assignments = []
        pilot1_assignments = []
        pilot2_assignments = []
        crew1_assignments = []
        crew2_assignments = []
        crew3_assignments = []

        for i in range(self.n_flights):
            base_idx = i * 6
            aircraft_assignments.append(genome[base_idx])
            pilot1_assignments.append(genome[base_idx + 1])
            pilot2_assignments.append(genome[base_idx + 2])
            crew1_assignments.append(genome[base_idx + 3])
            crew2_assignments.append(genome[base_idx + 4])
            crew3_assignments.append(genome[base_idx + 5])

        # Calculate objective function components
        total_revenue = sum(self.flights['revenue'])
        total_operating_cost = sum(self.aircraft['cost'][aircraft_assignments[i]] * self.flights["duration"][i] for i in range(self.n_flights))

        # Calculate personnel costs
        pilot_hours = [0] * self.n_pilots
        crew_hours = [0] * self.n_crew

        total_pilot_cost = 0
        total_crew_cost = 0

        for i in range(self.n_flights):
            flight_duration = self.flights['duration'][i]

            # Pilot costs and hours
            p1, p2 = pilot1_assignments[i], pilot2_assignments[i]
            pilot_hours[p1] += flight_duration
            pilot_hours[p2] += flight_duration
            total_pilot_cost += (self.pilots['salary_per_hour'][p1] +
                                 self.pilots['salary_per_hour'][p2]) * flight_duration

            # Crew costs and hours
            c1, c2, c3 = crew1_assignments[i], crew2_assignments[i], crew3_assignments[i]
            crew_hours[c1] += flight_duration
            crew_hours[c2] += flight_duration
            crew_hours[c3] += flight_duration
            total_crew_cost += (self.crew['salary_per_hour'][c1] +
                                self.crew['salary_per_hour'][c2] +
                                self.crew['salary_per_hour'][c3]) * flight_duration

        total_cost = total_operating_cost + total_pilot_cost + total_crew_cost
        base_objective = total_revenue - total_cost

        # Calculate constraint violations and penalties
        penalties = 0

        # Constraint: Average emission constraint (hard)
        total_emissions = sum(self.aircraft['emission_rate'][aircraft_assignments[i]] *
                              (self.flights['duration'][i] ** self.constraints["emission_exponent"])
                              for i in range(self.n_flights))
        avg_emissions = total_emissions / self.n_flights if self.n_flights > 0 else 0

        if avg_emissions > self.constraints['avg_emission_max']:
            penalties += 1000 * (avg_emissions - self.constraints['avg_emission_max'])

        # Constraint: Budget constraint (soft)
        if total_cost > self.constraints['budget']:
            budget_overrun = max(0, total_cost - self.constraints['budget'])
            alpha = self.constraints['lambda_budget']
            penalties += ((alpha + 1) * (budget_overrun)) / (alpha * budget_overrun + 1)

        # Constraint: Working hours constraints (soft)
        for p_idx in range(self.n_pilots):
            total_pilot_hours = self.pilots['logged_hours'][p_idx] + pilot_hours[p_idx]
            if total_pilot_hours > self.constraints['max_hours_pilot']:
                beta = self.constraints['lambda_work']
                pilot_hours_over = total_pilot_hours - self.constraints['max_hours_pilot']
                penalties +=  ((beta + 1) * pilot_hours_over) / (beta * pilot_hours_over + 1)

        for c_idx in range(self.n_crew):
            total_crew_hours = self.crew['logged_hours'][c_idx] + crew_hours[c_idx]
            if total_crew_hours > self.constraints['max_hours_crew']:
                beta = self.constraints['lambda_work']
                crew_hours_over = total_crew_hours - self.constraints['max_hours_crew']
                penalties += ((beta + 1) * crew_hours_over) / (beta * crew_hours_over + 1)

        # Hard constraint penalties for duplicates within flights
        for i in range(self.n_flights):
            # Check pilot duplicates
            pilots_for_flight = [pilot1_assignments[i], pilot2_assignments[i]]
            if len(set(pilots_for_flight)) != len(pilots_for_flight):
                penalties += 10000  # Heavy penalty for duplicate pilots
            # Check crew duplicates
            crew_for_flight = [crew1_assignments[i], crew2_assignments[i], crew3_assignments[i]]
            if len(set(crew_for_flight)) != len(crew_for_flight):
                penalties += 10000  # Heavy penalty for duplicate crew

        fitness = base_objective - penalties
        return fitness, {'avg_emissions': avg_emissions, 'total_cost': total_cost}


def custom_mutate(individual, mutation_rate=0.1):
    """Custom mutation operator that respects the problem structure"""
    genome = individual.genome[:]
    n_flights = len(genome) // 6

    for i in range(n_flights):
        if random.random() < mutation_rate:
            base_idx = i * 6

            # Mutate aircraft assignment
            if random.random() < 0.5:
                genome[base_idx] = random.randint(0, problem.n_aircraft - 1)

            # Mutate pilot assignments
            if random.random() < 0.5:
                new_pilots = random.sample(range(problem.n_pilots), min(2, problem.n_pilots))
                while len(new_pilots) < 2:
                    new_pilots.append(random.randint(0, problem.n_pilots - 1))
                genome[base_idx + 1:base_idx + 3] = new_pilots

            # Mutate crew assignments
            if random.random() < 0.5:
                new_crew = random.sample(range(problem.n_crew), min(3, problem.n_crew))
                while len(new_crew) < 3:
                    new_crew.append(random.randint(0, problem.n_crew - 1))
                genome[base_idx + 3:base_idx + 6] = new_crew

    return Individual(genome)


def custom_crossover(parent1, parent2):
    """Custom crossover that maintains flight structure"""
    genome1, genome2 = parent1.genome[:], parent2.genome[:]
    n_flights = len(genome1) // 6

    offspring1, offspring2 = genome1[:], genome2[:]

    for i in range(n_flights):
        if random.random() < 0.5:  # Swap entire flight assignments
            base_idx = i * 6
            offspring1[base_idx:base_idx + 6], offspring2[base_idx:base_idx + 6] = \
                offspring2[base_idx:base_idx + 6], offspring1[base_idx:base_idx + 6]

    return [Individual(offspring1), Individual(offspring2)]


# Example usage and problem setup
def run_flight_scheduling_optimization():
    """Run the genetic algorithm optimization"""

    # --- Problem Dimensions ---
    num_flights = 50
    num_aircraft = 80
    num_pilots = 120
    num_crew = 190

    # data
    flights_data, aircraft_data, pilot_data, crew_data = simulate_data(num_flights, num_aircraft, num_pilots, num_crew)

    constraints = {
        'budget': 500000,  # B
        'max_hours_pilot': 8,  # Hp
        'max_hours_crew': 8,  # Hc
        'emission_exponent': 1.5,
        'avg_emission_max': 2000,  # Eavg_max
        'lambda_budget': 0.5,  # Budget penalty weight
        'lambda_work': 0.5  # Work hours penalty weight
    }

    global problem
    problem = FlightSchedulingProblem(flights_data, aircraft_data, pilot_data, crew_data, constraints)

    # Create initial population
    def create_individual():
        genome = problem.create_individual()
        return Individual(genome, decoder=IdentityDecoder(), problem=problem)

    # Set up the evolutionary algorithm
    pop_size = 100
    generations = 5000

    # Create initial population
    population = [create_individual() for _ in range(pop_size)]

    # Evaluate initial population
    for individual in population:
        individual.fitness, cost_and_emission = problem.evaluate_fitness(individual)

    # Evolution parameters
    tournament_size = 3
    mutation_rate = 0.1
    crossover_rate = 0.8

    # Track best fitness and attributes over generations
    best_fitness_history = []
    cost_and_emission_history = []

    # Run evolution
    for generation in range(generations):
        # Selection
        parents = []
        for _ in range(pop_size):
            tournament = random.sample(population, tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)

        # Crossover and mutation
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                if random.random() < crossover_rate:
                    child1, child2 = custom_crossover(parents[i], parents[i + 1])
                else:
                    child1, child2 = parents[i], parents[i + 1]

                # Mutation
                child1 = custom_mutate(child1, mutation_rate)
                child2 = custom_mutate(child2, mutation_rate)

                offspring.extend([child1, child2])
            else:
                offspring.append(custom_mutate(parents[i], mutation_rate))

        attributes = []
        index = 0
        # Evaluate offspring
        for individual in offspring:
            individual.fitness, attr = problem.evaluate_fitness(individual)
            attributes.append(attr)

        # Replace population
        population = offspring[:pop_size]

        # Track progress
        best_fitness = max(ind.fitness for ind in population)
        best_index = max(
            enumerate(population),
            key=lambda x: x[1].fitness
        )

        best_fitness_history.append(best_fitness)
        cost_and_emission_history.append(attributes[best_index[0]])

        if generation % 10 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.2f}")

    # Return best solution
    best_individual = max(population, key=lambda x: x.fitness)

    print(f"\nOptimization completed!")
    print(f"Best fitness: {best_individual.fitness:.2f}")

    # Decode and display solution
    genome = best_individual.genome
    print(f"\nBest solution assignments:")
    for i in range(problem.n_flights):
        base_idx = i * 6
        aircraft = genome[base_idx]
        pilots = [genome[base_idx + 1], genome[base_idx + 2]]
        crew = [genome[base_idx + 3], genome[base_idx + 4], genome[base_idx + 5]]
        print(f"Flight {i + 1}: Aircraft {aircraft}, Pilots {pilots}, Crew {crew}")

    # Plot fitness evolution
    plt.figure(figsize=(10, 6))
    plt.plot(np.convolve(best_fitness_history, np.ones(100)/100, mode='valid'))
    plt.title('Best Fitness Evolution')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.show()

    # Plot cost evolution
    plt.figure(figsize=(10, 6))
    plt.plot([list(d.values())[1] for d in cost_and_emission_history])
    plt.axhline(y=constraints['budget'], color='red', linestyle='--', label='Budget')
    plt.title('Cost of flights over generations')
    plt.xlabel('Generation')
    plt.ylabel('Total Cost')
    plt.grid(True)
    plt.show()

    # Plot carbon emission evolution
    plt.figure(figsize=(10, 6))
    plt.plot([list(d.values())[0] for d in cost_and_emission_history])
    plt.axhline(y=constraints['avg_emission_max'], color='red', linestyle='--', label='Maximum allowed carbon emission')
    plt.title('Carbon emission of flights over generations')
    plt.xlabel('Generation')
    plt.ylabel('Average emission')
    plt.grid(True)
    plt.show()

    return best_individual, best_fitness_history


def print_solution(best_solution):
    # Print the best solution
    print("\nBest Solution Genome:", best_solution.genome)
    print("Best Solution Fitness:", best_solution.fitness)
    # Create a table for the results using prettytable
    print("\n📊 Optimized Flight Assignment Summary:")
    result_table = PrettyTable()
    result_table.field_names = ["Flight ID", "Aircraft ID", "Pilot 1 ID", "Pilot 2 ID", "Crew 1 ID", "Crew 2 ID",
                                "Crew 3 ID"]
    for i in range(problem.n_flights):
        base_idx = i * 6
        aircraft = genome[base_idx]
        pilot1 = genome[base_idx + 1]
        pilot2 = genome[base_idx + 2]
        crew1 = genome[base_idx + 3]
        crew2 = genome[base_idx + 4]
        crew3 = genome[base_idx + 5]
        result_table.add_row([i + 1, aircraft, pilot1, pilot2, crew1, crew2, crew3])
    print(result_table)
    # print average emission for the best solution
    total_emissions = sum(problem.aircraft['emission_rate'][genome[i * 6]] *
                          (problem.flights['duration'][i] ** problem.constraints["emission_exponent"])
                          for i in range(problem.n_flights))
    avg_emissions = total_emissions / problem.n_flights if problem.n_flights > 0 else 0
    print(f"\nAverage Emissions for Best Solution: {avg_emissions:.2f} gms of carbon per km")


# Run the optimization
if __name__ == "__main__":
    best_solution, fitness_history = run_flight_scheduling_optimization()

    genome = best_solution.genome
    print_solution(best_solution)

    # Visualize results
    visualize_solution(best_solution, problem, fitness_history)