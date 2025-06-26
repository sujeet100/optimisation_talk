import random
import pandas as pd
from datetime import datetime, timedelta
import numpy as np


class AirlineDataGenerator:
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)

        # Airport codes and their congestion patterns
        self.airports = {
            'JFK': {'congestion_base': 0.7, 'hub_size': 'large'},
            'LAX': {'congestion_base': 0.75, 'hub_size': 'large'},
            # 'ORD': {'congestion_base': 0.8, 'hub_size': 'large'},
            # 'ATL': {'congestion_base': 0.85, 'hub_size': 'large'},
            # 'DFW': {'congestion_base': 0.7, 'hub_size': 'large'},
            'DEN': {'congestion_base': 0.6, 'hub_size': 'medium'},
            'PHX': {'congestion_base': 0.55, 'hub_size': 'medium'},
            # 'LAS': {'congestion_base': 0.6, 'hub_size': 'medium'},
            # 'MIA': {'congestion_base': 0.65, 'hub_size': 'medium'},
            # 'SEA': {'congestion_base': 0.5, 'hub_size': 'medium'},
            # 'BOS': {'congestion_base': 0.7, 'hub_size': 'medium'},
            'MSP': {'congestion_base': 0.45, 'hub_size': 'small'},
            'DTW': {'congestion_base': 0.4, 'hub_size': 'small'},
            # 'CLT': {'congestion_base': 0.5, 'hub_size': 'small'},
            # 'PHL': {'congestion_base': 0.65, 'hub_size': 'medium'}
        }

        # Aircraft types with realistic specifications
        self.aircraft_types = {
            'A320': {'capacity': 180, 'range': 3300, 'fuel_consumption': 650, 'operating_cost': 4200, 'turnaround': 45},
            'A321': {'capacity': 220, 'range': 4000, 'fuel_consumption': 720, 'operating_cost': 4800, 'turnaround': 50},
            'B737-800': {'capacity': 189, 'range': 3115, 'fuel_consumption': 680, 'operating_cost': 4500,
                         'turnaround': 45},
            'B737 MAX': {'capacity': 210, 'range': 3550, 'fuel_consumption': 620, 'operating_cost': 4300,
                         'turnaround': 45},
            # 'A330': {'capacity': 335, 'range': 7400, 'fuel_consumption': 1800, 'operating_cost': 8500,
            #          'turnaround': 90},
            'B777': {'capacity': 396, 'range': 9700, 'fuel_consumption': 2200, 'operating_cost': 10200,
                     'turnaround': 100},
            # 'B787': {'capacity': 330, 'range': 8200, 'fuel_consumption': 1650, 'operating_cost': 8000,
            #          'turnaround': 85},
            'A350': {'capacity': 350, 'range': 8100, 'fuel_consumption': 1700, 'operating_cost': 8200, 'turnaround': 90}
        }

    def calculate_distance(self, origin, destination):
        """Simplified distance calculation between airports (in nautical miles)"""
        # This is a simplified model - in reality you'd use actual coordinates
        airport_coords = {
            'JFK': (40.6413, -73.7781), 'LAX': (33.9425, -118.4081), 'ORD': (41.9742, -87.9073),
            'ATL': (33.6407, -84.4277), 'DFW': (32.8998, -97.0403), 'DEN': (39.8561, -104.6737),
            'PHX': (33.4484, -112.0740), 'LAS': (36.0840, -115.1537), 'MIA': (25.7959, -80.2870),
            'SEA': (47.4502, -122.3088), 'BOS': (42.3656, -71.0096), 'MSP': (44.8848, -93.2223),
            'DTW': (42.2162, -83.3554), 'CLT': (35.2144, -80.9473), 'PHL': (39.8744, -75.2424)
        }

        if origin in airport_coords and destination in airport_coords:
            lat1, lon1 = airport_coords[origin]
            lat2, lon2 = airport_coords[destination]
            # Simplified distance calculation (not accurate but realistic for demo)
            distance = ((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) ** 0.5 * 69 * 0.868976  # Convert to nautical miles
            return max(100, int(distance))  # Minimum 100 nm
        return random.randint(200, 3000)

    def generate_flights(self, num_flights=100):
        """Generate flight data with observed attributes only"""
        flights = []

        for i in range(num_flights):
            origin = random.choice(list(self.airports.keys()))
            destination = random.choice([apt for apt in self.airports.keys() if apt != origin])

            distance = self.calculate_distance(origin, destination)

            # Flight duration based on distance (simplified)
            duration = max(60, int(distance * 0.15 + random.randint(-15, 30)))  # minutes

            # Time to departure (next 7 days)
            departure_time = datetime.now() + timedelta(
                days=random.randint(0, 6),
                hours=random.randint(5, 23),
                minutes=random.choice([0, 15, 30, 45])
            )

            # Airport congestion varies by time of day
            hour = departure_time.hour
            base_congestion = self.airports[origin]['congestion_base']
            if 6 <= hour <= 9 or 17 <= hour <= 20:  # Peak hours
                congestion = min(1.0, base_congestion + random.uniform(0.1, 0.3))
            else:
                congestion = max(0.1, base_congestion - random.uniform(0.1, 0.2))

            # Weather risk based on season and randomness
            month = departure_time.month
            if month in [12, 1, 2]:  # Winter
                weather_risk = random.uniform(0.3, 0.8)
            elif month in [6, 7, 8]:  # Summer (thunderstorms)
                weather_risk = random.uniform(0.2, 0.6)
            else:
                weather_risk = random.uniform(0.1, 0.4)

            # Flight priority (1=low, 2=medium, 3=high)
            priority = random.choices([1, 2, 3], weights=[0.3, 0.5, 0.2])[0]

            # Passenger demand based on route popularity and time
            route_popularity = 0.8 if self.airports[origin]['hub_size'] == 'large' and \
                                      self.airports[destination]['hub_size'] == 'large' else 0.5
            demand = min(1.0, route_popularity + random.uniform(-0.2, 0.3))

            # Estimated revenue based on distance, priority, and demand
            base_revenue_per_mile = 0.35 if distance < 500 else (0.25 if distance < 1500 else 0.18)
            revenue_multiplier = 1.0 + (priority - 2) * 0.3 + (demand - 0.5) * 0.4
            estimated_revenue = int(distance * base_revenue_per_mile * revenue_multiplier *
                                    random.uniform(0.8, 1.2) * 100)  # Average 100 passengers

            flights.append({
                'flight_id': f'FL{i + 1:04d}',
                'source_airport': origin,
                'destination_airport': destination,
                'time_to_departure': departure_time,
                'distance_nm': distance,
                'duration_minutes': duration,
                'flight_priority': priority,
                'passenger_demand': round(demand, 3),
                'estimated_revenue_usd': estimated_revenue,
                'airport_congestion': round(congestion, 3),
                'weather_risk_factor': round(weather_risk, 3)
            })

        return pd.DataFrame(flights)

    def generate_aircraft(self, num_aircraft=80):
        """Generate aircraft data with observed attributes only"""
        aircraft = []

        for i in range(num_aircraft):
            aircraft_type = random.choice(list(self.aircraft_types.keys()))
            specs = self.aircraft_types[aircraft_type]

            # Hours flown this week (0-70 hours, regulatory limit ~100)
            hours_this_week = random.uniform(0, 70)

            # Maintenance due time (hours until next maintenance)
            maintenance_due = random.randint(10, 500)

            # Fuel efficiency score (0-1, based on aircraft age and type)
            age_factor = random.uniform(0.8, 1.0)  # Newer aircraft are more efficient
            base_efficiency = 1.0 - (specs['fuel_consumption'] - 600) / 2000  # Normalize
            fuel_efficiency = min(1.0, max(0.3, base_efficiency * age_factor))

            # Current location
            current_airport = random.choice(list(self.airports.keys()))

            aircraft.append({
                'aircraft_id': f'AC{i + 1:03d}',
                'aircraft_type': aircraft_type,
                'passenger_capacity': specs['capacity'],
                'range_nm': specs['range'],
                'fuel_consumption_lbs_per_hour': specs['fuel_consumption'],
                'operating_cost_usd_per_hour': specs['operating_cost'],
                'maintenance_due_hours': maintenance_due,
                'hours_flown_this_week': round(hours_this_week, 1),
                'fuel_efficiency_score': round(fuel_efficiency, 3),
                'current_airport_location': current_airport,
                'turnaround_time_minutes': specs['turnaround']
            })

        return pd.DataFrame(aircraft)

    def generate_crew(self, num_crew=200):
        """Generate crew data with observed attributes only"""
        crew = []

        for i in range(num_crew):
            # Crew type
            crew_type = random.choices(['pilot', 'cabin_crew'], weights=[0.25, 0.75])[0]

            # Salary per hour (realistic airline industry rates)
            if crew_type == 'pilot':
                salary_per_hour = random.uniform(45, 120)  # $45-120/hour for pilots
            else:
                salary_per_hour = random.uniform(18, 35)  # $18-35/hour for cabin crew

            # Hours clocked this week (regulatory limits: pilots 100hrs/month, cabin crew similar)
            hours_clocked = random.uniform(0, 65)

            # Last rest duration (hours since last duty)
            last_rest = random.uniform(8, 48)  # FAA requires minimum 8-10 hours rest

            # Current location
            current_location = random.choice(list(self.airports.keys()))

            # Experience level (years of experience normalized to 0-1)
            experience_years = random.uniform(0.5, 25)
            experience_level = min(1.0, experience_years / 20)  # Cap at 20 years

            # Fatigue score (higher = more fatigued)
            # Based on hours worked and rest time
            fatigue_base = min(1.0, hours_clocked / 60)  # Normalize hours
            rest_factor = max(0, (24 - last_rest) / 24)  # Less rest = more fatigue
            fatigue_score = min(1.0, (fatigue_base + rest_factor) / 2 + random.uniform(-0.1, 0.1))

            # Availability (0=not available, 1=fully available)
            # Based on duty time regulations and fatigue
            if hours_clocked > 55 or last_rest < 10 or fatigue_score > 0.8:
                availability = random.uniform(0, 0.3)
            else:
                availability = random.uniform(0.7, 1.0)

            crew.append({
                'crew_id': f'CR{i + 1:04d}',
                'crew_type': crew_type,
                'availability_score': round(availability, 3),
                'hours_clocked_this_week': round(hours_clocked, 1),
                'last_rest_duration_hours': round(last_rest, 1),
                'current_location_airport': current_location,
                'experience_level': round(experience_level, 3),
                'salary_usd_per_hour': round(salary_per_hour, 2),
                'fatigue_score': round(fatigue_score, 3)
            })

        return pd.DataFrame(crew)

    def generate_all_data(self, num_flights=100, num_aircraft=50, num_crew=200):
        """Generate all datasets"""
        print("Generating airline scheduling data...")

        flights_df = self.generate_flights(num_flights)
        aircraft_df = self.generate_aircraft(num_aircraft)
        crew_df = self.generate_crew(num_crew)

        print(f"Generated {len(flights_df)} flights, {len(aircraft_df)} aircraft, {len(crew_df)} crew members")

        return flights_df, aircraft_df, crew_df


# Example usage
if __name__ == "__main__":
    generator = AirlineDataGenerator(seed=42)

    # Generate data
    flights, aircraft, crew = generator.generate_all_data(
        num_flights=100,
        num_aircraft=80,
        num_crew=450
    )

    # Display sample data
    print("\n=== FLIGHTS SAMPLE ===")
    print(flights.head())
    print(f"\nFlights columns: {list(flights.columns)}")

    print("\n=== AIRCRAFT SAMPLE ===")
    print(aircraft.head())
    print(f"\nAircraft columns: {list(aircraft.columns)}")

    print("\n=== CREW SAMPLE ===")
    print(crew.head())
    print(f"\nCrew columns: {list(crew.columns)}")

    # Save to CSV files
    flights.to_csv('flights_data.csv', index=False)
    aircraft.to_csv('aircraft_data.csv', index=False)
    crew.to_csv('crew_data.csv', index=False)

    print("\nData saved to CSV files: flights_data.csv, aircraft_data.csv, crew_data.csv")

    # Summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Average flight revenue: ${flights['estimated_revenue_usd'].mean():,.0f}")
    print(f"Average aircraft operating cost: ${aircraft['operating_cost_usd_per_hour'].mean():,.0f}/hour")
    print(f"Average pilot salary: ${crew[crew['crew_type'] == 'pilot']['salary_usd_per_hour'].mean():.2f}/hour")
    print(
        f"Average cabin crew salary: ${crew[crew['crew_type'] == 'cabin_crew']['salary_usd_per_hour'].mean():.2f}/hour")
    print(f"Flight distance range: {flights['distance_nm'].min()}-{flights['distance_nm'].max()} nm")
    print(
        f"Available crew: {len(crew[crew['availability_score'] > 0.7])}/{len(crew)} ({len(crew[crew['availability_score'] > 0.7]) / len(crew) * 100:.1f}%)")