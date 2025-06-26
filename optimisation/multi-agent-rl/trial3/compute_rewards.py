import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta


class DirectCSVRewardSystem:
    """
    Reward system using your exact CSV column names - no mapping or validation
    """

    def __init__(self,
                 flights_csv_path: str,
                 aircraft_csv_path: str,
                 crew_csv_path: str,
                 weekly_budget: float = 5000000,
                 max_emissions: float = 500000):

        self.weekly_budget = weekly_budget
        self.max_emissions = max_emissions

        # Load CSV data directly
        self.flights_data = pd.read_csv(flights_csv_path)
        self.aircraft_data = pd.read_csv(aircraft_csv_path)
        self.crew_data = pd.read_csv(crew_csv_path)

        # Create lookup dictionaries for fast access
        self.aircraft_lookup = self.aircraft_data.set_index('aircraft_id').to_dict('index')
        self.crew_lookup = self.crew_data.set_index('crew_id').to_dict('index')
        self.flights_lookup = self.flights_data.set_index('flight_id').to_dict('index')

        # Agent weights
        self.coordination_weights = {
            'base': 0.35,
            'fleet': 0.25,
            'crew': 0.25,
            'emissions': 0.15
        }

        print(
            f"✅ Loaded {len(self.flights_data)} flights, {len(self.aircraft_data)} aircraft, {len(self.crew_data)} crew members")

    def compute_rewards(self,
                        scheduled_flight_ids: List[str],
                        aircraft_assignments: Dict[str, str],
                        crew_assignments: Dict[str, List[str]]) -> Dict:
        """
        Compute rewards using actual CSV data with your exact column names
        """

        # Filter scheduled flights from full dataset
        scheduled_flights = self.flights_data[
            self.flights_data['flight_id'].isin(scheduled_flight_ids)
        ].copy()

        # Calculate individual agent rewards
        base_reward = self._calculate_base_reward(scheduled_flights, aircraft_assignments, crew_assignments)
        fleet_reward = self._calculate_fleet_reward(scheduled_flights, aircraft_assignments)
        crew_reward = self._calculate_crew_reward(scheduled_flights, crew_assignments)
        emissions_reward = self._calculate_emissions_reward(scheduled_flights, aircraft_assignments)

        # Calculate weighted system reward
        system_reward = (
                base_reward['total_reward'] * self.coordination_weights['base'] +
                fleet_reward['total_reward'] * self.coordination_weights['fleet'] +
                crew_reward['total_reward'] * self.coordination_weights['crew'] +
                emissions_reward['total_reward'] * self.coordination_weights['emissions']
        )

        return {
            'agent_rewards': {
                'base_agent': base_reward['total_reward'],
                'fleet_agent': fleet_reward['total_reward'],
                'crew_agent': crew_reward['total_reward'],
                'emissions_agent': emissions_reward['total_reward']
            },
            'system_reward': np.clip(system_reward, -2.0, 2.0),
            'detailed_rewards': {
                'base_agent': base_reward,
                'fleet_agent': fleet_reward,
                'crew_agent': crew_reward,
                'emissions_agent': emissions_reward
            },
            'operational_metrics': self._calculate_operational_metrics(
                scheduled_flights, aircraft_assignments, crew_assignments),
            'data_summary': {
                'total_flights_available': len(self.flights_data),
                'flights_scheduled': len(scheduled_flights),
                'aircraft_used': len(set(aircraft_assignments.values())),
                'crew_assigned': len(set([c for crew_list in crew_assignments.values() for c in crew_list]))
            }
        }

    def _calculate_base_reward(self, scheduled_flights: pd.DataFrame,
                               aircraft_assignments: Dict, crew_assignments: Dict) -> Dict:
        """Calculate base agent reward using your exact column names"""

        # 1. PROFITABILITY (40% weight)
        total_revenue = scheduled_flights['estimated_revenue_usd'].sum()
        total_operating_cost = self._calculate_actual_operating_cost(
            scheduled_flights, aircraft_assignments, crew_assignments)
        profit = total_revenue - total_operating_cost

        profit_normalized = np.tanh(profit / (self.weekly_budget * 0.2))
        profit_reward = profit_normalized * 0.4

        # 2. SCHEDULE COVERAGE (30% weight)
        coverage_ratio = len(scheduled_flights) / len(self.flights_data)
        coverage_normalized = np.clip(coverage_ratio * 2 - 1, -1, 1)
        coverage_reward = coverage_normalized * 0.3

        # 3. BUDGET UTILIZATION (20% weight)
        budget_utilization = total_operating_cost / self.weekly_budget
        if budget_utilization <= 0.8:
            budget_normalized = (budget_utilization / 0.8) * 2 - 1
        else:
            budget_normalized = max(-1, 2 - 2 * budget_utilization)
        budget_reward = budget_normalized * 0.2

        # 4. OPERATIONAL EFFICIENCY (10% weight)
        aircraft_efficiency = len(aircraft_assignments) / max(1, len(scheduled_flights))
        crew_efficiency = len(crew_assignments) / max(1, len(scheduled_flights))
        operational_score = (aircraft_efficiency + crew_efficiency) / 2
        operational_normalized = np.clip(operational_score * 2 - 1, -1, 1)
        operational_reward = operational_normalized * 0.1

        total_reward = profit_reward + coverage_reward + budget_reward + operational_reward

        return {
            'total_reward': np.clip(total_reward, -2.0, 2.0),
            'profit_reward': profit_reward,
            'coverage_reward': coverage_reward,
            'budget_reward': budget_reward,
            'operational_reward': operational_reward,
            'actual_profit_usd': profit,
            'actual_revenue_usd': total_revenue,
            'actual_operating_cost_usd': total_operating_cost,
            'coverage_ratio': coverage_ratio,
            'budget_utilization': budget_utilization
        }

    def _calculate_actual_operating_cost(self, flights: pd.DataFrame,
                                         aircraft_assignments: Dict, crew_assignments: Dict) -> float:
        """Calculate actual operating cost using your CSV data"""
        total_cost = 0

        for _, flight in flights.iterrows():
            flight_id = flight['flight_id']
            duration_hours = flight['duration_minutes'] / 60

            # Aircraft operating costs (from your CSV)
            if flight_id in aircraft_assignments:
                aircraft_id = aircraft_assignments[flight_id]
                if aircraft_id in self.aircraft_lookup:
                    aircraft = self.aircraft_lookup[aircraft_id]

                    # Direct operating cost from your CSV
                    operating_cost_per_hour = aircraft['operating_cost_usd_per_hour']
                    aircraft_cost = operating_cost_per_hour * duration_hours

                    # Add fuel cost
                    # fuel_consumption_lbs = aircraft['fuel_consumption_lbs_per_hour']
                    # fuel_cost = fuel_consumption_lbs * 2.5 * duration_hours  # $2.5 per lb

                    total_cost += aircraft_cost #+ fuel_cost

            # Crew salary costs (from your CSV)
            if flight_id in crew_assignments:
                crew_list = crew_assignments[flight_id]
                for crew_id in crew_list:
                    if crew_id in self.crew_lookup:
                        crew = self.crew_lookup[crew_id]
                        salary_per_hour = crew['salary_usd_per_hour']
                        total_cost += salary_per_hour * duration_hours

            # Airport and operational fees (based on your airport_congestion factor)
            # congestion_factor = flight.get('airport_congestion', 1.0)
            # base_airport_fee = 500  # Base airport fee per flight
            # airport_cost = base_airport_fee * (1 + congestion_factor)
            # total_cost += airport_cost

        return total_cost

    def _calculate_fleet_reward(self, scheduled_flights: pd.DataFrame,
                                aircraft_assignments: Dict) -> Dict:
        """Calculate fleet agent reward using your aircraft data"""

        # 1. AIRCRAFT UTILIZATION (40% weight)
        utilization_score = self._calculate_aircraft_utilization(scheduled_flights, aircraft_assignments)
        target_utilization = 0.75
        utilization_normalized = 1 - 2 * abs(utilization_score - target_utilization) / target_utilization
        utilization_normalized = np.clip(utilization_normalized, -1, 1)
        utilization_reward = utilization_normalized * 0.4

        # 2. AIRCRAFT-FLIGHT MATCHING (35% weight)
        matching_score = self._calculate_aircraft_flight_matching(scheduled_flights, aircraft_assignments)
        matching_normalized = matching_score * 2 - 1
        matching_reward = matching_normalized * 0.35

        # 3. MAINTENANCE EFFICIENCY (15% weight)
        maintenance_score = self._calculate_maintenance_efficiency(aircraft_assignments)
        maintenance_normalized = maintenance_score * 2 - 1
        maintenance_reward = maintenance_normalized * 0.15

        # 4. FLEET EFFICIENCY (10% weight)
        efficiency_score = self._calculate_fleet_efficiency(aircraft_assignments)
        efficiency_normalized = efficiency_score * 2 - 1
        efficiency_reward = efficiency_normalized * 0.1

        total_reward = utilization_reward + matching_reward + maintenance_reward + efficiency_reward

        return {
            'total_reward': np.clip(total_reward, -2.0, 2.0),
            'utilization_reward': utilization_reward,
            'matching_reward': matching_reward,
            'maintenance_reward': maintenance_reward,
            'efficiency_reward': efficiency_reward,
            'actual_utilization': utilization_score,
            'aircraft_assigned': len(aircraft_assignments),
            'unique_aircraft_used': len(set(aircraft_assignments.values()))
        }

    def _calculate_aircraft_utilization(self, flights: pd.DataFrame, assignments: Dict) -> float:
        """Calculate aircraft utilization using hours_flown_this_week"""
        if not assignments:
            return 0

        aircraft_hours = {}
        for _, flight in flights.iterrows():
            flight_id = flight['flight_id']
            if flight_id in assignments:
                aircraft_id = assignments[flight_id]
                duration_hours = flight['duration_minutes'] / 60
                aircraft_hours[aircraft_id] = aircraft_hours.get(aircraft_id, 0) + duration_hours

        # Add existing weekly hours from CSV
        for aircraft_id in aircraft_hours:
            if aircraft_id in self.aircraft_lookup:
                existing_hours = self.aircraft_lookup[aircraft_id]['hours_flown_this_week']
                aircraft_hours[aircraft_id] += existing_hours

        # Calculate utilization (max 112 hours per week)
        utilizations = [min(hours / 112, 1.5) for hours in aircraft_hours.values()]
        return np.mean(utilizations) if utilizations else 0

    def _calculate_aircraft_flight_matching(self, flights: pd.DataFrame, assignments: Dict) -> float:
        """Calculate aircraft-flight matching using your CSV specs"""
        if not assignments:
            return 0

        matching_scores = []
        for _, flight in flights.iterrows():
            flight_id = flight['flight_id']
            if flight_id in assignments:
                aircraft_id = assignments[flight_id]
                if aircraft_id in self.aircraft_lookup:
                    aircraft = self.aircraft_lookup[aircraft_id]

                    # Range adequacy
                    range_adequate = aircraft['range_nm'] >= flight['distance_nm']

                    # Capacity utilization
                    capacity_util = min(1.0, flight['passenger_demand'] / aircraft['passenger_capacity'])

                    # Fuel efficiency
                    fuel_efficiency = aircraft['fuel_efficiency_score']

                    # Weather and congestion consideration
                    weather_factor = 1.0 - flight['weather_risk_factor'] * 0.2
                    congestion_factor = 1.0 - flight['airport_congestion'] * 0.1

                    matching_score = (
                            range_adequate * 0.3 +
                            capacity_util * 0.3 +
                            fuel_efficiency * 0.2 +
                            weather_factor * 0.1 +
                            congestion_factor * 0.1
                    )
                    matching_scores.append(matching_score)

        return np.mean(matching_scores) if matching_scores else 0

    def _calculate_maintenance_efficiency(self, assignments: Dict) -> float:
        """Calculate maintenance efficiency using maintenance_due_hours"""
        if not assignments:
            return 1.0

        maintenance_scores = []
        used_aircraft = set(assignments.values())

        for aircraft_id in used_aircraft:
            if aircraft_id in self.aircraft_lookup:
                aircraft = self.aircraft_lookup[aircraft_id]
                maintenance_due = aircraft['maintenance_due_hours']

                if maintenance_due > 20:
                    maintenance_scores.append(1.0)
                elif maintenance_due > 0:
                    maintenance_scores.append(maintenance_due / 20)
                else:
                    maintenance_scores.append(0.0)  # Overdue

        return np.mean(maintenance_scores) if maintenance_scores else 1.0

    def _calculate_fleet_efficiency(self, assignments: Dict) -> float:
        """Calculate fleet efficiency"""
        if not assignments:
            return 0

        unique_aircraft = len(set(assignments.values()))
        total_assignments = len(assignments)

        # Efficiency = more flights per aircraft
        efficiency = min(1.0, total_assignments / unique_aircraft / 3)  # 3 flights per aircraft target
        return efficiency

    def _calculate_crew_reward(self, scheduled_flights: pd.DataFrame,
                               crew_assignments: Dict) -> Dict:
        """Calculate crew agent reward using your crew data"""

        # 1. CREW ALLOCATION ADEQUACY (40% weight)
        adequacy_score = self._calculate_crew_adequacy(crew_assignments)
        adequacy_normalized = adequacy_score * 2 - 1
        adequacy_reward = adequacy_normalized * 0.4

        # 2. CREW UTILIZATION & COST (30% weight)
        utilization_score = self._calculate_crew_utilization(scheduled_flights, crew_assignments)
        utilization_normalized = utilization_score * 2 - 1
        utilization_reward = utilization_normalized * 0.3

        # 3. FATIGUE & AVAILABILITY (20% weight)
        fatigue_score = self._calculate_crew_fatigue_management(crew_assignments)
        fatigue_normalized = fatigue_score * 2 - 1
        fatigue_reward = fatigue_normalized * 0.2

        # 4. EXPERIENCE MATCHING (10% weight)
        experience_score = self._calculate_crew_experience_matching(scheduled_flights, crew_assignments)
        experience_normalized = experience_score * 2 - 1
        experience_reward = experience_normalized * 0.1

        total_reward = adequacy_reward + utilization_reward + fatigue_reward + experience_reward

        return {
            'total_reward': np.clip(total_reward, -2.0, 2.0),
            'adequacy_reward': adequacy_reward,
            'utilization_reward': utilization_reward,
            'fatigue_reward': fatigue_reward,
            'experience_reward': experience_reward,
            'crew_assigned': len(set([c for crew_list in crew_assignments.values() for c in crew_list])),
            'flights_with_crew': len(crew_assignments)
        }

    def _calculate_crew_adequacy(self, crew_assignments: Dict) -> float:
        """Calculate crew adequacy using crew_type from your CSV"""
        if not crew_assignments:
            return 0

        adequacy_scores = []
        for flight_id, crew_list in crew_assignments.items():
            pilots = 0
            cabin_crew = 0

            for crew_id in crew_list:
                if crew_id in self.crew_lookup:
                    crew = self.crew_lookup[crew_id]
                    crew_type = crew['crew_type'].lower()
                    if 'pilot' in crew_type:
                        pilots += 1
                    elif 'cabin' in crew_type or 'attendant' in crew_type:
                        cabin_crew += 1

            # Standard requirements: 2 pilots, 3+ cabin crew
            pilot_adequacy = min(1.0, pilots / 2)
            cabin_adequacy = min(1.0, cabin_crew / 3)
            flight_adequacy = (pilot_adequacy + cabin_adequacy) / 2
            adequacy_scores.append(flight_adequacy)

        return np.mean(adequacy_scores) if adequacy_scores else 0

    def _calculate_crew_utilization(self, flights: pd.DataFrame, crew_assignments: Dict) -> float:
        """Calculate crew utilization using hours_clocked_this_week"""
        if not crew_assignments:
            return 0

        crew_hours = {}

        # Calculate additional hours from new assignments
        for _, flight in flights.iterrows():
            flight_id = flight['flight_id']
            if flight_id in crew_assignments:
                duration_hours = flight['duration_minutes'] / 60

                for crew_id in crew_assignments[flight_id]:
                    if crew_id in self.crew_lookup:
                        existing_hours = self.crew_lookup[crew_id]['hours_clocked_this_week']
                        total_hours = existing_hours + duration_hours
                        crew_hours[crew_id] = total_hours

        if not crew_hours:
            return 0

        # Calculate utilization efficiency (target: 40-50 hours per week)
        utilization_scores = []
        optimal_hours = 45

        for crew_id, hours in crew_hours.items():
            utilization = 1.0 - abs(hours - optimal_hours) / optimal_hours
            utilization_scores.append(max(0, utilization))

        return np.mean(utilization_scores)

    def _calculate_crew_fatigue_management(self, crew_assignments: Dict) -> float:
        """Calculate fatigue management using fatigue_score and availability_score"""
        if not crew_assignments:
            return 1.0

        assigned_crew = set()
        for crew_list in crew_assignments.values():
            assigned_crew.update(crew_list)

        fatigue_scores = []
        for crew_id in assigned_crew:
            if crew_id in self.crew_lookup:
                crew = self.crew_lookup[crew_id]

                # Fatigue score (lower is better)
                fatigue = crew['fatigue_score']
                fatigue_component = 1.0 - fatigue

                # Availability score (higher is better)
                availability = crew['availability_score']

                # Rest duration
                rest_hours = crew['last_rest_duration_hours']
                rest_component = min(1.0, rest_hours / 10)  # 10+ hours is optimal

                crew_score = (fatigue_component + availability + rest_component) / 3
                fatigue_scores.append(crew_score)

        return np.mean(fatigue_scores) if fatigue_scores else 1.0

    def _calculate_crew_experience_matching(self, flights: pd.DataFrame, crew_assignments: Dict) -> float:
        """Calculate experience matching using experience_level and flight_priority"""
        if not crew_assignments:
            return 0

        matching_scores = []
        for _, flight in flights.iterrows():
            flight_id = flight['flight_id']
            if flight_id in crew_assignments:
                flight_priority = flight['flight_priority']
                crew_list = crew_assignments[flight_id]

                experience_levels = []
                for crew_id in crew_list:
                    if crew_id in self.crew_lookup:
                        crew = self.crew_lookup[crew_id]
                        experience = crew['experience_level']
                        experience_levels.append(experience)

                if experience_levels:
                    avg_experience = np.mean(experience_levels)
                    # Higher priority flights should get more experienced crew
                    priority_normalized = flight_priority / 3.0  # Assuming max priority is 3
                    match_score = avg_experience * priority_normalized
                    matching_scores.append(match_score)

        return np.mean(matching_scores) if matching_scores else 0

    def _calculate_emissions_reward(self, scheduled_flights: pd.DataFrame,
                                    aircraft_assignments: Dict) -> Dict:
        """Calculate emissions reward using fuel_consumption_lbs_per_hour"""

        # Calculate actual emissions
        total_emissions = self._calculate_actual_emissions(scheduled_flights, aircraft_assignments)

        # 1. EMISSION MINIMIZATION (60% weight)
        emission_ratio = total_emissions / self.max_emissions if self.max_emissions > 0 else 0
        emission_normalized = max(-1, 2 - 2 * emission_ratio)
        emission_reward = emission_normalized * 0.6

        # 2. FUEL EFFICIENCY (40% weight)
        efficiency_score = self._calculate_fuel_efficiency(aircraft_assignments)
        efficiency_normalized = efficiency_score * 2 - 1
        efficiency_reward = efficiency_normalized * 0.4

        total_reward = emission_reward + efficiency_reward

        return {
            'total_reward': np.clip(total_reward, -2.0, 2.0),
            'emission_reward': emission_reward,
            'efficiency_reward': efficiency_reward,
            'actual_emissions_kg': total_emissions,
            'emission_ratio': emission_ratio,
            'emission_compliance': total_emissions <= self.max_emissions
        }

    def _calculate_actual_emissions(self, flights: pd.DataFrame, assignments: Dict) -> float:
        """Calculate actual emissions using your fuel consumption data"""
        total_emissions = 0
        fuel_to_co2_factor = 3.16  # kg CO2 per kg fuel

        for _, flight in flights.iterrows():
            flight_id = flight['flight_id']
            if flight_id in assignments:
                aircraft_id = assignments[flight_id]
                if aircraft_id in self.aircraft_lookup:
                    aircraft = self.aircraft_lookup[aircraft_id]

                    duration_hours = flight['duration_minutes'] / 60
                    fuel_consumption_lbs = aircraft['fuel_consumption_lbs_per_hour']
                    fuel_consumption_kg = fuel_consumption_lbs * 0.453592 * duration_hours

                    co2_emissions = fuel_consumption_kg * fuel_to_co2_factor
                    total_emissions += co2_emissions

        return total_emissions

    def _calculate_fuel_efficiency(self, assignments: Dict) -> float:
        """Calculate fuel efficiency using fuel_efficiency_score"""
        if not assignments:
            return 0.8

        efficiency_scores = []
        used_aircraft = set(assignments.values())

        for aircraft_id in used_aircraft:
            if aircraft_id in self.aircraft_lookup:
                aircraft = self.aircraft_lookup[aircraft_id]
                efficiency = aircraft['fuel_efficiency_score']
                efficiency_scores.append(efficiency)

        return np.mean(efficiency_scores) if efficiency_scores else 0.8

    def _calculate_operational_metrics(self, flights: pd.DataFrame,
                                       aircraft_assignments: Dict, crew_assignments: Dict) -> Dict:
        """Calculate operational metrics using your actual data"""

        # Financial metrics
        total_revenue = flights['estimated_revenue_usd'].sum()
        total_operating_cost = self._calculate_actual_operating_cost(flights, aircraft_assignments, crew_assignments)
        profit = total_revenue - total_operating_cost

        # Operational metrics
        total_distance = flights['distance_nm'].sum()
        total_duration = flights['duration_minutes'].sum()
        avg_flight_priority = flights['flight_priority'].mean()
        avg_passenger_demand = flights['passenger_demand'].mean()

        # Risk metrics
        avg_weather_risk = flights['weather_risk_factor'].mean()
        avg_congestion = flights['airport_congestion'].mean()

        # Resource metrics
        unique_aircraft = len(set(aircraft_assignments.values()))
        unique_crew = len(set([c for crew_list in crew_assignments.values() for c in crew_list]))

        return {
            'financial': {
                'total_revenue_usd': total_revenue,
                'total_operating_cost_usd': total_operating_cost,
                'profit_usd': profit,
                'profit_margin': profit / max(1, total_revenue),
                'avg_revenue_per_flight': total_revenue / max(1, len(flights))
            },
            'operational': {
                'flights_scheduled': len(flights),
                'total_distance_nm': total_distance,
                'total_duration_hours': total_duration / 60,
                'avg_flight_priority': avg_flight_priority,
                'avg_passenger_demand': avg_passenger_demand,
                'schedule_coverage_pct': len(flights) / len(self.flights_data) * 100
            },
            'risk_factors': {
                'avg_weather_risk': avg_weather_risk,
                'avg_airport_congestion': avg_congestion,
                'high_risk_flights': len(flights[flights['weather_risk_factor'] > 0.7])
            },
            'resource_utilization': {
                'aircraft_used': unique_aircraft,
                'crew_used': unique_crew,
                'flights_per_aircraft': len(flights) / max(1, unique_aircraft),
                'avg_crew_per_flight': unique_crew / max(1, len(flights))
            }
        }


# Easy wrapper function
def create_direct_csv_reward_system(flights_csv: str, aircraft_csv: str, crew_csv: str,
                                    weekly_budget: float = 5000000, max_emissions: float = 500000):
    """
    Create reward system using your exact CSV column names
    """
    return DirectCSVRewardSystem(
        flights_csv_path=flights_csv,
        aircraft_csv_path=aircraft_csv,
        crew_csv_path=crew_csv,
        weekly_budget=weekly_budget,
        max_emissions=max_emissions
    )


# Integration example
def integration_example():
    """Example integration with your environment"""

    example_code = '''
# In your AirlineSchedulingMAEnvironment:

class AirlineSchedulingMAEnvironment:
    def __init__(self, flights_csv, aircraft_csv, crew_csv):
        # Your existing initialization...

        # Initialize reward system with your CSV files
        self.reward_system = create_direct_csv_reward_system(
            flights_csv=flights_csv,
            aircraft_csv=aircraft_csv,
            crew_csv=crew_csv,
            weekly_budget=8000000,
            max_emissions=600000
        )

    def step(self, actions):
        # Process actions to get scheduling decisions
        scheduled_flight_ids = self.extract_scheduled_flights(actions)
        aircraft_assignments = self.extract_aircraft_assignments(actions) 
        crew_assignments = self.extract_crew_assignments(actions)

        # Calculate rewards using actual CSV data
        reward_results = self.reward_system.compute_rewards(
            scheduled_flight_ids=scheduled_flight_ids,
            aircraft_assignments=aircraft_assignments,
            crew_assignments=crew_assignments
        )

        # Extract agent rewards for MAPPO
        agent_rewards = reward_results['agent_rewards']

        return observations, agent_rewards, dones, infos
    '''

    print("🔧 DIRECT INTEGRATION EXAMPLE:")
    print(example_code)


# Test function
def test_direct_system():
    """Test with example scheduling scenario"""

    # Example usage
    try:
        reward_system = create_direct_csv_reward_system(
            flights_csv="flights.csv",
            aircraft_csv="aircraft.csv",
            crew_csv="crew.csv"
        )

        # Example scheduling
        scheduled_flights = ['FL0001', 'FL0002', 'FL0003']
        aircraft_assignments = {
            'FL0001': 'AC001',
            'FL0002': 'AC002',
            'FL0003': 'AC001'
        }
        crew_assignments = {
            'FL0001': ['CR0001', 'CR0002', 'CR0010', 'CR0011', 'CR0012'],
            'FL0002': ['CR0003', 'CR0004', 'CR0013', 'CR0014', 'CR0015'],
            'FL0003': ['CR0005', 'CR0006', 'CR0016', 'CR0017', 'CR0018']
        }

        results = reward_system.compute_rewards(
            scheduled_flight_ids=scheduled_flights,
            aircraft_assignments=aircraft_assignments,
            crew_assignments=crew_assignments
        )

        print("✅ Reward Results:")
        print(f"System Reward: {results['system_reward']:.4f}")
        for agent, reward in results['agent_rewards'].items():
            print(f"{agent}: {reward:.4f}")

        return results

    except Exception as e:
        print(f"Test failed: {e}")
        print("Make sure your CSV files exist with the exact column names specified")


if __name__ == "__main__":
    print("Direct CSV Reward System - No Mapping Required!")
    print("=" * 60)
    print("✅ Uses your exact column names")
    print("✅ No validation or mapping overhead")
    print("✅ Direct CSV access for maximum performance")
    print()

    integration_example()

    print("\n" + "=" * 60)
    print("READY TO USE:")
    print("Just replace the CSV file paths with your actual files!")