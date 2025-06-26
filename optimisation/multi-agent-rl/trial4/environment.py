import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import copy
from compute_rewards import DirectCSVRewardSystem


class AirlineSchedulingMAEnvironment:
    """
    Multi-Agent Reinforcement Learning Environment for Airline Scheduling

    4 Agents:
    - Base Agent: Overall scheduling and resource allocation decisions
    - Fleet Agent: Aircraft assignment and fleet optimization
    - Crew Agent: Crew scheduling and assignment
    - Emissions Agent: Environmental impact optimization
    """

    def __init__(self,
                 flights_csv_path: str,
                 aircraft_csv_path: str,
                 crew_csv_path: str,
                 max_flights_per_episode: int = 50,
                 weekly_budget: float = 5000000,
                 max_emissions: float = 500000,
                 episode_length: int = 100,
                 seed: Optional[int] = None):

        # Environment configuration
        self.max_flights_per_episode = max_flights_per_episode
        self.weekly_budget = weekly_budget
        self.max_emissions = max_emissions
        self.episode_length = episode_length
        self.current_step = 0

        # Load data
        self.flights_data = pd.read_csv(flights_csv_path)
        self.aircraft_data = pd.read_csv(aircraft_csv_path)
        self.crew_data = pd.read_csv(crew_csv_path)

        # FIXED: Add missing num_flights attribute
        self.num_flights = len(self.flights_data)

        # Calculate normalization constants from data for better scaling
        self.norm_constants = self._calculate_normalization_constants()

        # Initialize reward system
        self.reward_system = DirectCSVRewardSystem(
            flights_csv_path=flights_csv_path,
            aircraft_csv_path=aircraft_csv_path,
            crew_csv_path=crew_csv_path,
            weekly_budget=weekly_budget,
            max_emissions=max_emissions
        )

        # Agent names
        self.agents = ['base_agent', 'fleet_agent', 'crew_agent', 'emissions_agent']
        self.num_agents = len(self.agents)

        # Create lookup dictionaries for fast access
        self.flights_lookup = self.flights_data.set_index('flight_id').to_dict('index')
        self.aircraft_lookup = self.aircraft_data.set_index('aircraft_id').to_dict('index')
        self.crew_lookup = self.crew_data.set_index('crew_id').to_dict('index')

        # State tracking
        self.scheduled_flights = set()
        self.aircraft_assignments = {}  # flight_id -> aircraft_id
        self.crew_assignments = {}  # flight_id -> [crew_ids]
        self.available_flights = list(self.flights_data['flight_id'].values)
        self.available_aircraft = list(self.aircraft_data['aircraft_id'].values)
        self.available_crew = list(self.crew_data['crew_id'].values)

        # Resource utilization tracking
        self.aircraft_hours = {aid: 0 for aid in self.available_aircraft}
        self.crew_hours = {cid: 0 for cid in self.available_crew}
        self.total_cost = 0
        self.total_emissions = 0

        # Episode flights tracking
        self.current_episode_flights = []

        # Define action and observation spaces
        self._setup_spaces()

        # Random seed
        if seed is not None:
            np.random.seed(seed)

        print(f"✅ Environment initialized with {len(self.flights_data)} flights, "
              f"{len(self.aircraft_data)} aircraft, {len(self.crew_data)} crew members")
        print(f"📊 Normalization constants calculated from data")

    def use_simple_action_spaces(self):
        """Switch to simplified action spaces for frameworks that don't support complex spaces"""
        self.action_spaces = self.simple_action_spaces.copy()
        print("🔄 Switched to simplified action spaces")

    def use_box_action_spaces(self):
        """Switch to Box action spaces for continuous control"""
        self.action_spaces = {
            'base_agent': spaces.Box(low=0.0, high=1.0, shape=(len(self.available_flights),), dtype=np.float32),
            'fleet_agent': spaces.Box(low=0.0, high=1.0,
                                      shape=(self.max_flights_per_episode, len(self.available_aircraft)),
                                      dtype=np.float32),
            'crew_agent': spaces.Box(low=0.0, high=1.0, shape=(self.max_flights_per_episode, len(self.available_crew)),
                                     dtype=np.float32),
            'emissions_agent': spaces.Box(low=0.0, high=1.0, shape=(self.max_flights_per_episode,), dtype=np.float32)
        }
        print("🔄 Switched to Box (continuous) action spaces")

    def _calculate_normalization_constants(self) -> Dict[str, float]:
        """Calculate normalization constants from actual data for better scaling"""
        return {
            'max_revenue': self.flights_data['estimated_revenue_usd'].max(),
            'max_distance': self.flights_data['distance_nm'].max(),
            'max_duration': self.flights_data['duration_minutes'].max(),
            'max_passengers': self.flights_data['passenger_demand'].max(),
            'max_priority': self.flights_data['flight_priority'].max(),
            'max_aircraft_capacity': self.aircraft_data['passenger_capacity'].max(),
            'max_aircraft_range': self.aircraft_data['range_nm'].max(),
            'max_operating_cost': self.aircraft_data['operating_cost_usd_per_hour'].max(),
            'max_experience': self.crew_data['experience_level'].max(),
            'max_salary': self.crew_data['salary_usd_per_hour'].max()
        }

    def _setup_spaces(self):
        """Setup action and observation spaces for each agent"""

        # Action spaces for each agent (all discrete for MARL compatibility)
        self.action_spaces = {
            # Base Agent: Schedule flights (binary for each available flight)
            'base_agent': spaces.MultiBinary(len(self.available_flights)),

            # Fleet Agent: Aircraft assignment (categorical for each flight)
            'fleet_agent': spaces.MultiDiscrete([len(self.available_aircraft) + 1] * self.max_flights_per_episode),

            # Crew Agent: Crew assignment (up to 6 crew members per flight)
            'crew_agent': spaces.MultiDiscrete([len(self.available_crew) + 1] * (self.max_flights_per_episode * 6)),

            # Emissions Agent: Fuel efficiency preferences (discrete levels: low=0, medium=1, high=2)
            'emissions_agent': spaces.MultiDiscrete([3] * self.max_flights_per_episode)
        }

        # Alternative: Simplified action spaces for some MARL frameworks
        self.simple_action_spaces = {
            'base_agent': spaces.Discrete(2 ** min(10, len(self.available_flights))),  # Simplified binary encoding
            'fleet_agent': spaces.Discrete(len(self.available_aircraft) + 1),
            'crew_agent': spaces.Discrete((len(self.available_crew) + 1) ** 3),  # Simplified to 3 crew max
            'emissions_agent': spaces.Discrete(3)  # Low, Medium, High efficiency preference
        }

        # FIXED: Calculate observation dimensions more carefully
        # Base agent observation components
        flight_features = min(self.max_flights_per_episode, len(self.available_flights)) * 8
        aircraft_features = len(self.available_aircraft) * 4
        crew_features = len(self.available_crew) * 3
        global_features = 10
        obs_dim_base = flight_features + aircraft_features + crew_features + global_features

        # Fleet agent observation components
        aircraft_details = len(self.available_aircraft) * 6
        flight_requirements = min(self.max_flights_per_episode, len(self.available_flights)) * 5
        fleet_metrics = 8
        obs_dim_fleet = aircraft_details + flight_requirements + fleet_metrics

        # Crew agent observation components
        crew_details = len(self.available_crew) * 5
        flight_crew_requirements = min(self.max_flights_per_episode, len(self.available_flights)) * 4
        crew_metrics = 6
        obs_dim_crew = crew_details + flight_crew_requirements + crew_metrics

        # Emissions agent observation components
        aircraft_efficiency = len(self.available_aircraft) * 3
        flight_environmental = min(self.max_flights_per_episode, len(self.available_flights)) * 3
        emissions_metrics = 5
        obs_dim_emissions = aircraft_efficiency + flight_environmental + emissions_metrics

        self.observation_spaces = {
            'base_agent': spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim_base,), dtype=np.float32),
            'fleet_agent': spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim_fleet,), dtype=np.float32),
            'crew_agent': spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim_crew,), dtype=np.float32),
            'emissions_agent': spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim_emissions,), dtype=np.float32)
        }

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset environment to initial state"""
        if seed is not None:
            np.random.seed(seed)

        # Reset state
        self.current_step = 0
        self.current_flight_idx = 0  # For simplified action processing
        self.scheduled_flights = set()
        self.aircraft_assignments = {}
        self.crew_assignments = {}
        self.total_cost = 0
        self.total_emissions = 0

        # Reset resource utilization
        self.aircraft_hours = {aid: self.aircraft_lookup[aid]['hours_flown_this_week']
                               for aid in self.available_aircraft}
        self.crew_hours = {cid: self.crew_lookup[cid]['hours_clocked_this_week']
                           for cid in self.available_crew}

        # FIXED: Sample subset of flights for this episode and ensure they exist
        available_flight_ids = list(self.flights_data['flight_id'].values)
        episode_size = min(self.max_flights_per_episode, len(available_flight_ids))
        episode_flights = np.random.choice(
            available_flight_ids,
            size=episode_size,
            replace=False
        )
        self.current_episode_flights = [fid for fid in episode_flights if fid in self.flights_lookup]

        # Get initial observations
        observations = self._get_observations()
        info = self._get_info()

        return observations, info

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict]:
        """Execute one step in the environment"""

        self.current_step += 1

        # Process actions from each agent
        self._process_actions(actions)

        # Calculate rewards
        rewards = self._calculate_rewards()

        # Check if episode is done
        terminated = self._check_terminated()
        truncated = self._check_truncated()

        # Get new observations
        observations = self._get_observations()
        info = self._get_info()

        # Create done dictionary for each agent
        dones_terminated = {agent: terminated for agent in self.agents}
        dones_truncated = {agent: truncated for agent in self.agents}

        return observations, rewards, dones_terminated, dones_truncated, info

    def _process_actions(self, actions: Dict[str, np.ndarray]):
        """Process actions from all agents"""

        # Check if we're using simplified action spaces
        if isinstance(self.action_spaces['base_agent'], spaces.Discrete):
            self._process_simple_actions(actions)
        else:
            self._process_complex_actions(actions)

    def _process_complex_actions(self, actions: Dict[str, np.ndarray]):
        """Process actions for complex action spaces (MultiDiscrete, MultiBinary)"""

        # Base Agent: Flight scheduling decisions
        base_action = actions['base_agent']
        new_scheduled_flights = []

        # FIXED: Only process flights that exist in current episode
        action_length = min(len(base_action), len(self.current_episode_flights))

        for i in range(action_length):
            should_schedule = base_action[i]
            flight_id = self.current_episode_flights[i]

            if should_schedule and flight_id not in self.scheduled_flights:
                new_scheduled_flights.append(flight_id)
                self.scheduled_flights.add(flight_id)

        # Fleet Agent: Aircraft assignments
        fleet_action = actions['fleet_agent']
        for i, aircraft_idx in enumerate(fleet_action):
            if i < len(new_scheduled_flights) and aircraft_idx > 0:
                flight_id = new_scheduled_flights[i]
                aircraft_idx_adj = aircraft_idx - 1  # Adjust for 0-index (0 means no assignment)
                if aircraft_idx_adj < len(self.available_aircraft):
                    aircraft_id = self.available_aircraft[aircraft_idx_adj]
                    if self._can_assign_aircraft(flight_id, aircraft_id):
                        self.aircraft_assignments[flight_id] = aircraft_id

        # Crew Agent: Crew assignments
        crew_action = actions['crew_agent']
        crew_action_reshaped = crew_action.reshape(self.max_flights_per_episode, 6)

        for i, crew_assignments_for_flight in enumerate(crew_action_reshaped):
            if i < len(new_scheduled_flights):
                flight_id = new_scheduled_flights[i]
                assigned_crew = []

                for crew_idx in crew_assignments_for_flight:
                    if crew_idx > 0:  # 0 means no assignment
                        crew_idx_adj = crew_idx - 1
                        if crew_idx_adj < len(self.available_crew):
                            crew_id = self.available_crew[crew_idx_adj]
                            if self._can_assign_crew(flight_id, crew_id):
                                assigned_crew.append(crew_id)

                if assigned_crew:
                    self.crew_assignments[flight_id] = assigned_crew

        # Emissions Agent: Influence aircraft selection based on efficiency preferences
        emissions_action = actions['emissions_agent']
        self._apply_emissions_preferences(emissions_action)

        # Update resource utilization
        self._update_resource_utilization()

    def _process_simple_actions(self, actions: Dict[str, np.ndarray]):
        """Process actions for simplified action spaces (single Discrete per agent)"""

        # For simplified actions, we process one flight at a time
        if not hasattr(self, 'current_flight_idx'):
            self.current_flight_idx = 0

        if self.current_flight_idx >= len(self.current_episode_flights):
            return

        current_flight = self.current_episode_flights[self.current_flight_idx]

        # Base Agent: Schedule this flight or not
        base_action = actions['base_agent']
        if base_action == 1 and current_flight not in self.scheduled_flights:
            self.scheduled_flights.add(current_flight)

            # Fleet Agent: Choose aircraft
            fleet_action = actions['fleet_agent']
            if fleet_action > 0 and fleet_action <= len(self.available_aircraft):
                aircraft_id = self.available_aircraft[fleet_action - 1]
                if self._can_assign_aircraft(current_flight, aircraft_id):
                    self.aircraft_assignments[current_flight] = aircraft_id

            # Crew Agent: Simplified crew assignment (decode action to crew selection)
            crew_action = actions['crew_agent']
            crew_assignments = self._decode_crew_action(crew_action, current_flight)
            if crew_assignments:
                self.crew_assignments[current_flight] = crew_assignments

            # Emissions Agent: Apply preference to current assignment
            emissions_action = actions['emissions_agent']
            if current_flight in self.aircraft_assignments:
                self._apply_single_emissions_preference(current_flight, emissions_action)

        self.current_flight_idx += 1
        self._update_resource_utilization()

    def _decode_crew_action(self, action: int, flight_id: str) -> List[str]:
        """Decode simplified crew action to actual crew assignments"""
        # Simple heuristic: assign 2 pilots + 3 cabin crew based on action
        available_pilots = [cid for cid in self.available_crew
                            if 'pilot' in self.crew_lookup[cid]['crew_type'].lower()]
        available_cabin = [cid for cid in self.available_crew
                           if 'cabin' in self.crew_lookup[cid]['crew_type'].lower()]

        assigned_crew = []

        # Use action to select crew (simplified)
        action_mod = action % (len(available_pilots) + len(available_cabin))

        # Assign pilots
        for i, pilot_id in enumerate(available_pilots[:2]):
            if self._can_assign_crew(flight_id, pilot_id):
                assigned_crew.append(pilot_id)

        # Assign cabin crew
        for i, cabin_id in enumerate(available_cabin[:3]):
            if self._can_assign_crew(flight_id, cabin_id):
                assigned_crew.append(cabin_id)

        return assigned_crew

    def _apply_single_emissions_preference(self, flight_id: str, preference: int):
        """Apply emissions preference for a single flight"""
        if flight_id in self.aircraft_assignments:
            current_aircraft = self.aircraft_assignments[flight_id]

            if preference == 2:  # High efficiency
                best_aircraft = self._find_most_efficient_aircraft(flight_id)
                if best_aircraft and best_aircraft != current_aircraft:
                    if self._can_assign_aircraft(flight_id, best_aircraft):
                        self.aircraft_assignments[flight_id] = best_aircraft
            elif preference == 0:  # Low efficiency (cost focus)
                cheapest_aircraft = self._find_cheapest_aircraft(flight_id)
                if cheapest_aircraft and cheapest_aircraft != current_aircraft:
                    if self._can_assign_aircraft(flight_id, cheapest_aircraft):
                        self.aircraft_assignments[flight_id] = cheapest_aircraft

    def _can_assign_aircraft(self, flight_id: str, aircraft_id: str) -> bool:
        """Check if aircraft can be assigned to flight"""
        if flight_id not in self.flights_lookup or aircraft_id not in self.aircraft_lookup:
            return False

        flight = self.flights_lookup[flight_id]
        aircraft = self.aircraft_lookup[aircraft_id]

        # Check range capability
        if aircraft['range_nm'] < flight['distance_nm']:
            return False

        # Check maintenance status
        if aircraft['maintenance_due_hours'] <= 0:
            return False

        # Check hour limits (assuming max 112 hours per week)
        flight_duration = flight['duration_minutes'] / 60
        if self.aircraft_hours[aircraft_id] + flight_duration > 112:
            return False

        return True

    def _can_assign_crew(self, flight_id: str, crew_id: str) -> bool:
        """Check if crew member can be assigned to flight"""
        if flight_id not in self.flights_lookup or crew_id not in self.crew_lookup:
            return False

        crew = self.crew_lookup[crew_id]
        flight = self.flights_lookup[flight_id]

        # Check availability
        if crew['availability_score'] < 0.5:
            return False

        # Check fatigue
        if crew['fatigue_score'] > 0.8:
            return False

        # Check hour limits (max 50 hours per week for crew)
        flight_duration = flight['duration_minutes'] / 60
        if self.crew_hours[crew_id] + flight_duration > 50:
            return False

        return True

    def _apply_emissions_preferences(self, emissions_preferences: np.ndarray):
        """Apply emissions agent preferences to aircraft assignments"""
        # Convert discrete actions to preference levels: 0=low, 1=medium, 2=high efficiency preference
        scheduled_flights = list(self.scheduled_flights)

        for i, preference_level in enumerate(emissions_preferences):
            if i < len(scheduled_flights):
                flight_id = scheduled_flights[i]
                if flight_id in self.aircraft_assignments:
                    current_aircraft = self.aircraft_assignments[flight_id]

                    # Apply preference based on discrete level
                    if preference_level == 2:  # High efficiency preference
                        best_aircraft = self._find_most_efficient_aircraft(flight_id)
                        if best_aircraft and best_aircraft != current_aircraft:
                            if self._can_assign_aircraft(flight_id, best_aircraft):
                                self.aircraft_assignments[flight_id] = best_aircraft
                    elif preference_level == 0:  # Low efficiency preference (cost focus)
                        cheapest_aircraft = self._find_cheapest_aircraft(flight_id)
                        if cheapest_aircraft and cheapest_aircraft != current_aircraft:
                            if self._can_assign_aircraft(flight_id, cheapest_aircraft):
                                self.aircraft_assignments[flight_id] = cheapest_aircraft
                    # preference_level == 1 (medium) keeps current assignment

    def _find_most_efficient_aircraft(self, flight_id: str) -> Optional[str]:
        """Find most fuel-efficient aircraft for a flight"""
        if flight_id not in self.flights_lookup:
            return None

        flight = self.flights_lookup[flight_id]
        best_aircraft = None
        best_efficiency = 0

        for aircraft_id in self.available_aircraft:
            if self._can_assign_aircraft(flight_id, aircraft_id):
                aircraft = self.aircraft_lookup[aircraft_id]
                if aircraft['fuel_efficiency_score'] > best_efficiency:
                    best_efficiency = aircraft['fuel_efficiency_score']
                    best_aircraft = aircraft_id

        return best_aircraft

    def _find_cheapest_aircraft(self, flight_id: str) -> Optional[str]:
        """Find most cost-effective aircraft for a flight"""
        if flight_id not in self.flights_lookup:
            return None

        flight = self.flights_lookup[flight_id]
        best_aircraft = None
        lowest_cost = float('inf')

        for aircraft_id in self.available_aircraft:
            if self._can_assign_aircraft(flight_id, aircraft_id):
                aircraft = self.aircraft_lookup[aircraft_id]
                operating_cost = aircraft['operating_cost_usd_per_hour']
                if operating_cost < lowest_cost:
                    lowest_cost = operating_cost
                    best_aircraft = aircraft_id

        return best_aircraft

    def _update_resource_utilization(self):
        """Update resource utilization tracking"""
        # Reset and recalculate aircraft hours
        self.aircraft_hours = {aid: self.aircraft_lookup[aid]['hours_flown_this_week']
                               for aid in self.available_aircraft}

        # Reset and recalculate crew hours
        self.crew_hours = {cid: self.crew_lookup[cid]['hours_clocked_this_week']
                           for cid in self.available_crew}

        # Add hours from current assignments
        for flight_id, aircraft_id in self.aircraft_assignments.items():
            if flight_id in self.flights_lookup:
                flight = self.flights_lookup[flight_id]
                duration = flight['duration_minutes'] / 60
                self.aircraft_hours[aircraft_id] += duration

        # Update crew hours
        for flight_id, crew_list in self.crew_assignments.items():
            if flight_id in self.flights_lookup:
                flight = self.flights_lookup[flight_id]
                duration = flight['duration_minutes'] / 60
                for crew_id in crew_list:
                    self.crew_hours[crew_id] += duration

    def _calculate_rewards(self) -> Dict[str, float]:
        """Calculate rewards using the reward system"""
        if not self.scheduled_flights:
            # No flights scheduled, small penalty
            return {agent: -0.1 for agent in self.agents}

        try:
            # Use the reward system to calculate rewards
            reward_results = self.reward_system.compute_rewards(
                scheduled_flight_ids=list(self.scheduled_flights),
                aircraft_assignments=self.aircraft_assignments,
                crew_assignments=self.crew_assignments
            )

            # Extract individual agent rewards
            agent_rewards = reward_results['agent_rewards']

            # Add step penalty to encourage efficiency
            step_penalty = -0.01 * (self.current_step / self.episode_length)

            rewards = {}
            for agent in self.agents:
                base_reward = agent_rewards.get(agent, 0.0)
                rewards[agent] = base_reward + step_penalty

            return rewards

        except Exception as e:
            print(f"Error calculating rewards: {e}")
            return {agent: -1.0 for agent in self.agents}

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for each agent"""
        observations = {}

        # Base Agent Observation
        obs_base = []

        # Flight information (normalized using actual data statistics)
        max_flights_to_process = min(self.max_flights_per_episode, len(self.current_episode_flights))

        for i in range(max_flights_to_process):
            flight_id = self.current_episode_flights[i]
            if flight_id in self.flights_lookup:
                flight = self.flights_lookup[flight_id]
                obs_base.extend([
                    flight['estimated_revenue_usd'] / self.norm_constants['max_revenue'],
                    flight['distance_nm'] / self.norm_constants['max_distance'],
                    flight['duration_minutes'] / self.norm_constants['max_duration'],
                    flight['passenger_demand'] / self.norm_constants['max_passengers'],
                    flight['flight_priority'] / self.norm_constants['max_priority'],
                    flight['weather_risk_factor'],  # Already 0-1
                    flight['airport_congestion'],  # Already 0-1
                    1.0 if flight_id in self.scheduled_flights else 0.0  # Binary
                ])
            else:
                obs_base.extend([0] * 8)

        # Aircraft status
        for aircraft_id in self.available_aircraft:
            aircraft = self.aircraft_lookup[aircraft_id]
            obs_base.extend([
                self.aircraft_hours[aircraft_id] / 112,  # Utilization
                aircraft['maintenance_due_hours'] / 100,
                aircraft['fuel_efficiency_score'],
                1.0 if any(aid == aircraft_id for aid in self.aircraft_assignments.values()) else 0.0
            ])

        # Crew status
        for crew_id in self.available_crew:
            crew = self.crew_lookup[crew_id]
            obs_base.extend([
                self.crew_hours[crew_id] / 50,  # Utilization
                crew['availability_score'],
                crew['fatigue_score']
            ])

        # Global metrics
        obs_base.extend([
            len(self.scheduled_flights) / max(1, len(self.current_episode_flights)),
            self.total_cost / self.weekly_budget,
            self.total_emissions / self.max_emissions,
            len(self.aircraft_assignments) / max(1, len(self.scheduled_flights)),
            len(self.crew_assignments) / max(1, len(self.scheduled_flights)),
            self.current_step / self.episode_length,
            np.mean([self.aircraft_hours[aid] for aid in self.available_aircraft]) / 112,
            np.mean([self.crew_hours[cid] for cid in self.available_crew]) / 50,
            len(set(self.aircraft_assignments.values())) / max(1, len(self.available_aircraft)),
            len(set([c for crew_list in self.crew_assignments.values() for c in crew_list])) / max(1,
                                                                                                   len(self.available_crew))
        ])

        observations['base_agent'] = np.array(obs_base, dtype=np.float32)

        # Fleet Agent Observation
        obs_fleet = []

        # Aircraft details
        for aircraft_id in self.available_aircraft:
            aircraft = self.aircraft_lookup[aircraft_id]
            obs_fleet.extend([
                aircraft['passenger_capacity'] / 400,
                aircraft['range_nm'] / 10000,
                aircraft['operating_cost_usd_per_hour'] / 10000,
                aircraft['fuel_efficiency_score'],
                self.aircraft_hours[aircraft_id] / 112,
                aircraft['maintenance_due_hours'] / 100
            ])

        # Flight requirements
        for i in range(max_flights_to_process):
            flight_id = self.current_episode_flights[i]
            if flight_id in self.flights_lookup:
                flight = self.flights_lookup[flight_id]
                obs_fleet.extend([
                    flight['distance_nm'] / 5000,
                    flight['passenger_demand'] / 300,
                    flight['duration_minutes'] / 600,
                    flight['weather_risk_factor'],
                    1.0 if flight_id in self.scheduled_flights else 0.0
                ])
            else:
                obs_fleet.extend([0] * 5)

        # Fleet metrics
        obs_fleet.extend([
            len(self.aircraft_assignments) / max(1, len(self.scheduled_flights)),
            len(set(self.aircraft_assignments.values())) / max(1, len(self.available_aircraft)),
            np.mean([self.aircraft_hours[aid] for aid in self.available_aircraft]) / 112,
            sum(1 for aid in self.available_aircraft if self.aircraft_lookup[aid]['maintenance_due_hours'] > 20) / len(
                self.available_aircraft),
            len(self.scheduled_flights) / max(1, len(self.current_episode_flights)),
            self.current_step / self.episode_length,
            0,  # Reserved
            0  # Reserved
        ])

        observations['fleet_agent'] = np.array(obs_fleet, dtype=np.float32)

        # Crew Agent Observation
        obs_crew = []

        # Crew details
        for crew_id in self.available_crew:
            crew = self.crew_lookup[crew_id]
            obs_crew.extend([
                crew['experience_level'] / 5,
                crew['availability_score'],
                crew['fatigue_score'],
                self.crew_hours[crew_id] / 50,
                crew['salary_usd_per_hour'] / 200
            ])

        # Flight crew requirements
        for i in range(max_flights_to_process):
            flight_id = self.current_episode_flights[i]
            if flight_id in self.flights_lookup:
                flight = self.flights_lookup[flight_id]
                obs_crew.extend([
                    flight['flight_priority'] / 3,
                    flight['duration_minutes'] / 600,
                    1.0 if flight_id in self.scheduled_flights else 0.0,
                    len(self.crew_assignments.get(flight_id, [])) / 6
                ])
            else:
                obs_crew.extend([0] * 4)

        # Crew metrics
        obs_crew.extend([
            len(self.crew_assignments) / max(1, len(self.scheduled_flights)),
            len(set([c for crew_list in self.crew_assignments.values() for c in crew_list])) / max(1,
                                                                                                   len(self.available_crew)),
            np.mean([self.crew_hours[cid] for cid in self.available_crew]) / 50,
            np.mean([self.crew_lookup[cid]['fatigue_score'] for cid in self.available_crew]),
            np.mean([self.crew_lookup[cid]['availability_score'] for cid in self.available_crew]),
            self.current_step / self.episode_length
        ])

        observations['crew_agent'] = np.array(obs_crew, dtype=np.float32)

        # Emissions Agent Observation
        obs_emissions = []

        # Aircraft efficiency
        for aircraft_id in self.available_aircraft:
            aircraft = self.aircraft_lookup[aircraft_id]
            obs_emissions.extend([
                aircraft['fuel_efficiency_score'],
                aircraft['fuel_consumption_lbs_per_hour'] / 2000,
                1.0 if any(aid == aircraft_id for aid in self.aircraft_assignments.values()) else 0.0
            ])

        # Flight environmental impact
        for i in range(max_flights_to_process):
            flight_id = self.current_episode_flights[i]
            if flight_id in self.flights_lookup:
                flight = self.flights_lookup[flight_id]
                obs_emissions.extend([
                    flight['distance_nm'] / 5000,
                    flight['duration_minutes'] / 600,
                    1.0 if flight_id in self.scheduled_flights else 0.0
                ])
            else:
                obs_emissions.extend([0] * 3)

        # Emissions metrics
        obs_emissions.extend([
            self.total_emissions / self.max_emissions,
            len(self.scheduled_flights) / max(1, len(self.current_episode_flights)),
            np.mean([self.aircraft_lookup[aid]['fuel_efficiency_score']
                     for aid in set(self.aircraft_assignments.values())]) if self.aircraft_assignments else 0,
            self.current_step / self.episode_length,
            0  # Reserved
        ])

        observations['emissions_agent'] = np.array(obs_emissions, dtype=np.float32)

        return observations

    def _check_terminated(self) -> bool:
        """Check if episode should terminate (success condition)"""
        # Calculate current costs and emissions
        self._update_costs_and_emissions()

        # Terminate if budget exceeded or emissions exceeded
        if self.total_cost > self.weekly_budget * 1.2:
            return True
        if self.total_emissions > self.max_emissions * 1.2:
            return True

        # Terminate if all feasible flights are scheduled
        feasible_flights = len([f for f in self.current_episode_flights
                                if f in self.flights_lookup])
        if len(self.scheduled_flights) >= feasible_flights * 0.9:
            return True

        return False

    def _check_truncated(self) -> bool:
        """Check if episode should be truncated (time limit)"""
        return self.current_step >= self.episode_length

    def _update_costs_and_emissions(self):
        """Update current total costs and emissions"""
        if not self.scheduled_flights:
            self.total_cost = 0
            self.total_emissions = 0
            return

        try:
            # Get scheduled flights data
            scheduled_flight_data = self.flights_data[
                self.flights_data['flight_id'].isin(self.scheduled_flights)
            ]

            # Calculate costs using reward system
            self.total_cost = self.reward_system._calculate_actual_operating_cost(
                scheduled_flight_data, self.aircraft_assignments, self.crew_assignments
            )

            # Calculate emissions using reward system
            self.total_emissions = self.reward_system._calculate_actual_emissions(
                scheduled_flight_data, self.aircraft_assignments
            )
        except:
            # Fallback to simple calculation
            self.total_cost = len(self.scheduled_flights) * 1000  # Simple estimate
            self.total_emissions = len(self.scheduled_flights) * 100  # Simple estimate

    def _get_info(self) -> Dict:
        """Get additional information about the environment state"""
        try:
            # Update costs and emissions first
            self._update_costs_and_emissions()

            # Calculate current metrics
            scheduled_flight_data = self.flights_data[
                self.flights_data['flight_id'].isin(self.scheduled_flights)
            ]

            total_revenue = scheduled_flight_data['estimated_revenue_usd'].sum() if len(
                scheduled_flight_data) > 0 else 0

            # FIXED: Return info structure that matches what training expects
            base_info = {
                'step': self.current_step,
                'scheduled_flights': len(self.scheduled_flights),
                'available_flights': len(self.current_episode_flights),
                'aircraft_assigned': len(self.aircraft_assignments),
                'crew_assigned': len(self.crew_assignments),
                'total_revenue': total_revenue,
                'current_emissions': self.total_emissions,
                'budget_utilization': self.total_cost / self.weekly_budget,
                'emissions_ratio': self.total_emissions / self.max_emissions if self.max_emissions > 0 else 0,
                'episode_progress': self.current_step / self.episode_length,
                # Additional metrics for training logs
                'scheduled_flights_count': len(self.scheduled_flights),
                'emission_utilization': self.total_emissions / self.max_emissions if self.max_emissions > 0 else 0
            }

            # Return info per agent for compatibility with training
            return {agent: base_info.copy() for agent in self.agents}

        except Exception as e:
            # Fallback info
            fallback_info = {
                'step': self.current_step,
                'scheduled_flights_count': len(self.scheduled_flights),
                'budget_utilization': 0,
                'emission_utilization': 0
            }
            return {agent: fallback_info.copy() for agent in self.agents}

    def render(self, mode='human'):
        """Render the environment state"""
        if mode == 'human':
            print(f"\n=== Step {self.current_step} ===")
            print(f"Scheduled Flights: {len(self.scheduled_flights)}/{len(self.current_episode_flights)}")
            print(f"Aircraft Assignments: {len(self.aircraft_assignments)}")
            print(f"Crew Assignments: {len(self.crew_assignments)}")
            print(f"Budget Utilization: {self.total_cost / self.weekly_budget:.2%}")
            print(f"Emissions Ratio: {self.total_emissions / self.max_emissions:.2%}")

    def close(self):
        """Clean up environment"""
        pass


# Wrapper for compatibility with different MARL libraries
class MAEnvironmentWrapper:
    """Wrapper to make environment compatible with different MARL frameworks"""

    def __init__(self, env: AirlineSchedulingMAEnvironment):
        self.env = env
        self.agents = env.agents
        self.num_agents = env.num_agents
        self.observation_spaces = env.observation_spaces
        self.action_spaces = env.action_spaces

    def reset(self, seed=None):
        return self.env.reset(seed)

    def step(self, actions):
        return self.env.step(actions)

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()


# Factory function for easy environment creation
def make_airline_env(flights_csv: str,
                     aircraft_csv: str,
                     crew_csv: str,
                     action_space_type: str = "discrete",  # "discrete", "simple", or "box"
                     **kwargs) -> AirlineSchedulingMAEnvironment:
    """Factory function to create airline scheduling environment

    Args:
        action_space_type: Type of action spaces to use
            - "discrete": MultiDiscrete/MultiBinary (default, most compatible)
            - "simple": Single Discrete per agent (for simple MARL algorithms)
            - "box": Continuous Box spaces (for continuous control)
    """
    env = AirlineSchedulingMAEnvironment(
        flights_csv_path=flights_csv,
        aircraft_csv_path=aircraft_csv,
        crew_csv_path=crew_csv,
        **kwargs
    )

    if action_space_type == "simple":
        env.use_simple_action_spaces()
    elif action_space_type == "box":
        env.use_box_action_spaces()
    # "discrete" is default

    return env


# Example usage and testing
if __name__ == "__main__":
    print("Testing fixed environment...")

    # Test 1: Default discrete action spaces (most compatible)
    print("\n1. Testing discrete action spaces:")
    try:
        env = make_airline_env(
            flights_csv="flights_data.csv",
            aircraft_csv="aircraft_data.csv",
            crew_csv="crew_data.csv",
            action_space_type="discrete",
            max_flights_per_episode=20,
            episode_length=50
        )

        print("Environment created successfully!")
        print(f"Agents: {env.agents}")
        print(f"Action space types: {[type(space).__name__ for space in env.action_spaces.values()]}")

        # Test reset
        obs, info = env.reset()
        print(f"Reset successful, got observations for {len(obs)} agents")

        # Verify observation dimensions match spaces
        for agent in env.agents:
            obs_shape = obs[agent].shape
            expected_shape = env.observation_spaces[agent].shape
            print(f"{agent}: obs shape {obs_shape}, expected {expected_shape}, match: {obs_shape == expected_shape}")

        # Test random actions
        actions = {}
        for agent in env.agents:
            actions[agent] = env.action_spaces[agent].sample()

        obs, rewards, terminated, truncated, info = env.step(actions)
        print(f"Step successful, rewards: {rewards}")
        print(f"Info structure: {list(info.keys())}")

        print("✅ Environment test passed!")

    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        import traceback

        traceback.print_exc()