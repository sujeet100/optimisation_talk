import pulp
import pandas as pd

import pandas as pd
from prettytable import PrettyTable

flights_df = pd.DataFrame([
    {"flight_id": "F1", "source": "DEL", "destination": "BOM", "distance_km": 1150, "duration_hr": 2, "required_model": "Airbus A320"},
    {"flight_id": "F2", "source": "BOM", "destination": "BLR", "distance_km": 980, "duration_hr": 2, "required_model": "Boeing 737"},
    {"flight_id": "F3", "source": "BLR", "destination": "HYD", "distance_km": 500, "duration_hr": 1, "required_model": "ATR 72"},
    {"flight_id": "F4", "source": "HYD", "destination": "MAA", "distance_km": 610, "duration_hr": 1, "required_model": "Bombardier Q400"},
])

aircraft_df = pd.DataFrame([
    {"aircraft_id": "A1", "model": "Airbus A320", "fuel_cost_per_hour": 5000},
    {"aircraft_id": "A2", "model": "Airbus A320", "fuel_cost_per_hour": 5200},
    {"aircraft_id": "A3", "model": "Boeing 737", "fuel_cost_per_hour": 5500},
    {"aircraft_id": "A4", "model": "Boeing 737", "fuel_cost_per_hour": 5400},
    {"aircraft_id": "A5", "model": "ATR 72", "fuel_cost_per_hour": 3000},
    {"aircraft_id": "A6", "model": "ATR 72", "fuel_cost_per_hour": 3100},
    {"aircraft_id": "A7", "model": "Bombardier Q400", "fuel_cost_per_hour": 3200},
    {"aircraft_id": "A8", "model": "Bombardier Q400", "fuel_cost_per_hour": 3300},
    {"aircraft_id": "A9", "model": "Airbus A321", "fuel_cost_per_hour": 5800},
    {"aircraft_id": "A10", "model": "Airbus A321", "fuel_cost_per_hour": 6000},
    {"aircraft_id": "A11", "model": "Airbus A320", "fuel_cost_per_hour": 5100},
    {"aircraft_id": "A12", "model": "Boeing 737", "fuel_cost_per_hour": 5600},
    {"aircraft_id": "A13", "model": "ATR 72", "fuel_cost_per_hour": 3050},
    {"aircraft_id": "A14", "model": "Bombardier Q400", "fuel_cost_per_hour": 3250},
    {"aircraft_id": "A15", "model": "Airbus A321", "fuel_cost_per_hour": 5950},
    {"aircraft_id": "A16", "model": "Airbus A320", "fuel_cost_per_hour": 5150},
    {"aircraft_id": "A17", "model": "Boeing 737", "fuel_cost_per_hour": 5700},
    {"aircraft_id": "A18", "model": "ATR 72", "fuel_cost_per_hour": 2950},
    {"aircraft_id": "A19", "model": "Bombardier Q400", "fuel_cost_per_hour": 3350},
    {"aircraft_id": "A20", "model": "Airbus A321", "fuel_cost_per_hour": 6100},
])

crew_df = pd.DataFrame([
    {"crew_id": "C1", "name": "Alice", "role": "Pilot", "remaining_hours": 8},
    {"crew_id": "C2", "name": "Bob", "role": "Co-pilot", "remaining_hours": 7},
    {"crew_id": "C3", "name": "Charlie", "role": "Attendant", "remaining_hours": 6},
    {"crew_id": "C4", "name": "David", "role": "Attendant", "remaining_hours": 5},
    {"crew_id": "C5", "name": "Eva", "role": "Pilot", "remaining_hours": 4},
    {"crew_id": "C6", "name": "Frank", "role": "Co-pilot", "remaining_hours": 6},
    {"crew_id": "C7", "name": "Grace", "role": "Attendant", "remaining_hours": 8},
    {"crew_id": "C8", "name": "Hannah", "role": "Pilot", "remaining_hours": 6},
    {"crew_id": "C9", "name": "Ivan", "role": "Co-pilot", "remaining_hours": 7},
    {"crew_id": "C10", "name": "Jack", "role": "Attendant", "remaining_hours": 5},
    {"crew_id": "C11", "name": "Karen", "role": "Attendant", "remaining_hours": 6},
    {"crew_id": "C12", "name": "Leo", "role": "Pilot", "remaining_hours": 8},
    {"crew_id": "C13", "name": "Maya", "role": "Co-pilot", "remaining_hours": 4},
    {"crew_id": "C14", "name": "Nina", "role": "Attendant", "remaining_hours": 6},
    {"crew_id": "C15", "name": "Oscar", "role": "Attendant", "remaining_hours": 5},
    {"crew_id": "C16", "name": "Paul", "role": "Pilot", "remaining_hours": 7},
    {"crew_id": "C17", "name": "Quinn", "role": "Co-pilot", "remaining_hours": 6},
    {"crew_id": "C18", "name": "Rita", "role": "Attendant", "remaining_hours": 8},
    {"crew_id": "C19", "name": "Sam", "role": "Attendant", "remaining_hours": 7},
    {"crew_id": "C20", "name": "Tina", "role": "Attendant", "remaining_hours": 6},
])


# Create the MILP problem
prob = pulp.LpProblem("Flight_Assignment_Problem", pulp.LpMinimize)

# --------- Decision Variables ---------

# x[f, a] = 1 if aircraft 'a' is assigned to flight 'f'
x = pulp.LpVariable.dicts("AssignAircraft",
    ((f, a) for f in flights_df['flight_id'] for a in aircraft_df['aircraft_id']),
    cat='Binary')

# y[f, c] = 1 if crew 'c' is assigned to flight 'f'
y = pulp.LpVariable.dicts("AssignCrew",
    ((f, c) for f in flights_df['flight_id'] for c in crew_df['crew_id']),
    cat='Binary')

# --------- Objective Function: Minimize fuel cost ---------

fuel_cost = pulp.lpSum(
    x[f, a] * flights_df.loc[flights_df.flight_id == f, 'duration_hr'].values[0] *
    aircraft_df.loc[aircraft_df.aircraft_id == a, 'fuel_cost_per_hour'].values[0]
    for f in flights_df['flight_id']
    for a in aircraft_df['aircraft_id']
)
prob += fuel_cost

# --------- Constraints ---------

# 1. Each flight is assigned exactly one aircraft of the required model
for f in flights_df['flight_id']:
    required_model = flights_df.loc[flights_df.flight_id == f, 'required_model'].values[0]
    matching_aircrafts = aircraft_df[aircraft_df.model == required_model]['aircraft_id'].tolist()
    prob += pulp.lpSum(x[f, a] for a in matching_aircrafts) == 1, f"AircraftAssignment_{f}"

# 2. Each flight must have:
#    - At least 1 pilot
#    - At least 1 co-pilot
#    - At least 1 attendant
for f in flights_df['flight_id']:
    prob += pulp.lpSum(y[f, c] for c in crew_df[crew_df.role == "Pilot"]['crew_id']) >= 1, f"PilotReq_{f}"
    prob += pulp.lpSum(y[f, c] for c in crew_df[crew_df.role == "Co-pilot"]['crew_id']) >= 1, f"CoPilotReq_{f}"
    prob += pulp.lpSum(y[f, c] for c in crew_df[crew_df.role == "Attendant"]['crew_id']) >= 1, f"AttendantReq_{f}"

# 3. Crew flight hour limits
for c in crew_df['crew_id']:
    max_hours = crew_df.loc[crew_df.crew_id == c, 'remaining_hours'].values[0]
    prob += pulp.lpSum(
        y[f, c] * flights_df.loc[flights_df.flight_id == f, 'duration_hr'].values[0]
        for f in flights_df['flight_id']
    ) <= max_hours, f"CrewHourLimit_{c}"

# Ensure each aircraft is used only once
for a in aircraft_df['aircraft_id']:
    prob += pulp.lpSum(x[f, a] for f in flights_df['flight_id']) <= 1, f"AircraftUsedOnce_{a}"

# Ensure each crew member is used only once
for c in crew_df['crew_id']:
    prob += pulp.lpSum(y[f, c] for f in flights_df['flight_id']) <= 1, f"CrewUsedOnce_{c}"

# Optional: prevent assigning crew to flights requiring unavailable roles (not needed here, all roles present)

# --------- Solve the Problem ---------

prob.solve()

# --------- Results ---------

print("\nStatus:", pulp.LpStatus[prob.status])
print("Total Fuel Cost:", pulp.value(prob.objective))

# Assigned Aircraft
# Create a table for the results
result_table = PrettyTable()
result_table.field_names = ["Flight ID", "Aircraft ID", "Crew IDs", "Fuel Cost"]

total_cost = 0.0

for f in flights_df['flight_id']:
    # Get assigned aircraft
    assigned_aircraft = next((a for a in aircraft_df['aircraft_id'] if pulp.value(x[f, a]) == 1), None)

    # Get assigned crew
    assigned_crew = [c for c in crew_df['crew_id'] if pulp.value(y[f, c]) == 1]

    # Calculate cost
    duration = flights_df.loc[flights_df.flight_id == f, 'duration_hr'].values[0]
    cost = 0.0
    if assigned_aircraft:
        fuel_cost_per_hour = aircraft_df.loc[aircraft_df.aircraft_id == assigned_aircraft, 'fuel_cost_per_hour'].values[
            0]
        cost = fuel_cost_per_hour * duration
        total_cost += cost

    # Add row to table
    result_table.add_row([f, assigned_aircraft, ", ".join(assigned_crew), f"{cost:.2f}"])

# Print summary
print("\n📊 Optimized Flight Assignment Summary:")
print(result_table)
print(f"\n💰 Total Fuel Cost: {total_cost:.2f}")