import numpy as np
import pandas as pd
import json

np.random.seed(42)  # for reproducibility

# --- Configuration ---
num_flights = 50
num_aircraft = 80
num_pilots = 120
num_crew = 190

# --- Sets ---
F = list(range(1, num_flights + 1))
A = list(range(1, num_aircraft + 1))
P = list(range(1, num_pilots + 1))
C = list(range(1, num_crew + 1))

# --- Parameters ---

# Flight particulars
Ri = np.random.randint(12000, 40000, size=num_flights) # Revenue from flight
Di = np.random.randint(2, 10, size=num_flights)  # Duration of flight in hours
Pi = np.random.random(size=num_flights) # flight priority score

# --- Synthetic Weather Data ---
wind_speed = np.random.uniform(0, 30, size=num_flights)         # knots
turbulence = np.random.uniform(0, 1.0, size=num_flights)        # index [0-1]
humidity = np.random.uniform(30, 90, size=num_flights) / 100.0  # relative humidity [0.3-0.9]

# Aircraft cost and revenue
Cj = np.random.randint(800, 1500, size=num_aircraft)  # Operating cost
Fj = np.random.uniform(2.5,6.0, size=num_aircraft) # Fuel efficiency (km/kg of fuel)
Ej = np.random.uniform(150, 200, size=num_aircraft)  # Emission base rate: gms of carbon per km

# Pilot and crew salaries and working time
Sm_p = np.random.randint(200, 400, size=num_pilots)  # Pilot salaries
Hm_p = np.random.uniform(0, 8, size=num_pilots) # Already logged working hours for pilots
Sn_c = np.random.randint(100, 200, size=num_crew)    # Crew salaries
Hn_c = np.random.uniform(0, 8, size=num_crew) # Already logged working hours for crew

# --- Weather Impact Factors ---
phi = 1 + 0.05 * wind_speed + 0.1 * turbulence
omega = 1 + 0.03 * wind_speed + 0.07 * humidity

# Max working hours
Hp = 8  # Pilot daily max hours
Hc = 8  # Crew daily max hours

# Emission average cap
Eavg_max = 1800

# Budget cap
B_cap = 1000000

# --- Export to Pandas DataFrames ---
df_aircraft = pd.DataFrame({
    "AircraftID": A,
    "op_cost_per_km": Cj,
    "fuel_efficiency_km_per_kg": Fj,
    "carbon_emission_gm_per_km": Ej
})

df_pilots = pd.DataFrame({
    "PilotID": P,
    "salary_pilot": Sm_p,
    "logged_hours_pilot": Hm_p
})

df_crew = pd.DataFrame({
    "CrewID": C,
    "salary_crew": Sn_c,
    "logged_hours_crew": Hn_c
})

df_flights = pd.DataFrame({
    "FlightID": F,
    "flight_duration": Di,
    "flight_revenue": Ri,
    "flight_priority": Pi,
    "wind_speed_expected": wind_speed,
    "turbulence_expected": turbulence,
    "humidity_expected": humidity,
    "weather_based_fuel_degradation_factor": phi,
    "weather_based_emission_amplification_factor": omega

})

opt_params = {
    "num_flights": num_flights,
    "num_aircraft": num_aircraft,
    "num_pilots": num_pilots,
    "num_crew": num_crew,
    "max_hours_pilot": Hp,
    "max_hours_crew": Hc,
    "max_allowed_avg_emission": Eavg_max,
    "budget_cap": B_cap
}

# --- Save to CSV ---
df_aircraft.to_csv("./data_sim/aircraft_data.csv", index=False)
df_pilots.to_csv("./data_sim/pilot_data.csv", index=False)
df_crew.to_csv("./data_sim/crew_data.csv", index=False)
df_flights.to_csv("./data_sim/flight_data.csv", index=False)
with open("./data_sim/opt_params.json", "w") as f:
    json.dump(opt_params, f)

# --- Display Summary ---
print("Data generation complete.")
print(f"Flights: {num_flights}, Aircraft: {num_aircraft}, Pilots: {num_pilots}, Crew: {num_crew}")
