import numpy as np
import pandas as pd
import json

def simulate_data(num_flights, num_aircraft, num_pilots, num_crew):
    np.random.seed(42)  # for reproducibility

    # Flight particulars
    Ri = np.random.randint(12000, 40000, size=num_flights) # Revenue from flight
    Di = np.random.randint(2, 10, size=num_flights)  # Duration of flight in hours

    # Aircraft cost and revenue
    Cj = np.random.randint(800, 1500, size=num_aircraft)  # Operating cost
    Ej = np.random.uniform(150, 200, size=num_aircraft)  # Emission base rate: gms of carbon per km

    # Pilot and crew salaries and working time
    Sm_p = np.random.randint(200, 400, size=num_pilots)  # Pilot salaries
    Hm_p = np.random.uniform(0, 8, size=num_pilots) # Already logged working hours for pilots
    Sn_c = np.random.randint(100, 200, size=num_crew)    # Crew salaries
    Hn_c = np.random.uniform(0, 8, size=num_crew) # Already logged working hours for crew

    flights_data = {
        'revenue':  Ri,
        'duration': Di
    }

    aircraft_data = {
        'cost':  Cj,
        'emission_rate': Ej
    }

    pilot_data = {
        'salary_per_hour': Sm_p,
        'logged_hours':  Hm_p
    }

    crew_data = {
        'salary_per_hour':  Sn_c,
        'logged_hours': Hn_c
    }

    return flights_data, aircraft_data, pilot_data, crew_data