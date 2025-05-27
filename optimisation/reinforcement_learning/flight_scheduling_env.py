import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FlightSchedulingEnv(gym.Env):
    def __init__(self, data, emission_limit=1000, budget_cap=50000):
        super(FlightSchedulingEnv, self).__init__()
        self.data = data
        self.num_flights = data["num_flights"]
        self.num_aircraft = data["num_aircraft"]
        self.num_pilots = data["num_pilots"]
        self.num_crew = data["num_crew"]
        self.budget_cap = data["budget_cap"]
        self.emission_limit = data["max_allowed_avg_emission"]

        # Action space: [w_i, aircraft, pilot1, pilot2, crew1, crew2, crew3] * num_flights
        self.action_space = spaces.MultiDiscrete([
            2,  # w_i: 0 or 1 (schedule flight)
            self.num_aircraft,  # aircraft index
            self.num_pilots,  # pilot 1 index
            self.num_pilots,  # pilot 2 index
            self.num_crew,  # crew 1 index
            self.num_crew,  # crew 2 index
            self.num_crew  # crew 3 index
        ])

        # Observation space: flight features + budget + pilot hours + crew hours + emissions
        obs_dim = (
            self.num_flights * 6    # per flight: w_i, Ri, Pi, Di, phi, omega
            + 1                    # normalized remaining budget
            + self.num_pilots       # normalized pilot hours left
            + self.num_crew         # normalized crew hours left
            + 1                    # normalized emissions
        )
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.state = np.zeros(obs_dim, dtype=np.float32)
        self.reset()

    def step(self, action):
        # Make this a multi-step environment
        if not hasattr(self, 'current_flight'):
            self.current_flight = 0
            self.total_revenue = 0
            self.total_cost = 0
            self.total_emissions = 0
            self.pilot_hours = np.array(self.data["logged_hours_pilot"]).copy()
            self.crew_hours = np.array(self.data["logged_hours_crew"]).copy()
            self.used_pilots = set()
            self.used_crew = set()
            self.scheduled_flights = np.zeros(self.num_flights)  # Track scheduled flights

        # Get action for current flight
        flight_idx = self.current_flight
        # Parse action (ensure it's exactly 7 elements)
        if isinstance(action, np.ndarray):
            if action.shape == (7,):
                w_i, a_idx, p1_idx, p2_idx, c1_idx, c2_idx, c3_idx = action
            else:
                raise ValueError(f"Expected action shape (7,), got {action.shape}")
        else:
            action_array = np.array(action)
            if action_array.size == 7:
                w_i, a_idx, p1_idx, p2_idx, c1_idx, c2_idx, c3_idx = action_array.flatten()
            else:
                raise ValueError(f"Expected 7 action elements, got {action_array.size}")

        # Convert to integers for indexing
        w_i = int(w_i)
        a_idx = int(a_idx)
        p1_idx = int(p1_idx)
        p2_idx = int(p2_idx)
        c1_idx = int(c1_idx)
        c2_idx = int(c2_idx)
        c3_idx = int(c3_idx)

        # Get flight data
        Ri = self.data["flight_revenue"][flight_idx]
        Pi = self.data["flight_priority"][flight_idx]
        Di = self.data["flight_duration"][flight_idx]
        Cj = self.data["op_cost_per_km"][a_idx]
        Ej = self.data["carbon_emission_gm_per_km"][a_idx]
        phi = self.data["weather_based_fuel_degradation_factor"][flight_idx]
        omega = self.data["weather_based_emission_amplification_factor"][flight_idx]
        Sm_p = self.data["salary_pilot"]
        Sn_c = self.data["salary_crew"]
        Hp = self.data["max_hours_pilot"]
        Hc = self.data["max_hours_crew"]

        reward = 0

        if w_i == 1:  # Schedule this flight
            self.scheduled_flights[flight_idx] = 1
            # Calculate base revenue
            flight_revenue = Ri * (1 + 0.3 * Pi)

            # Constraint violations with smooth penalties instead of hard cuts
            constraint_penalty = 0

            # Pilot assignment constraints (smooth penalty)
            if len({p1_idx, p2_idx}) < 2:
                constraint_penalty += 0.3 * flight_revenue  # 30% revenue penalty
            if p1_idx in self.used_pilots or p2_idx in self.used_pilots:
                constraint_penalty += 0.2 * flight_revenue  # 20% revenue penalty

            # Crew assignment constraints (smooth penalty)
            if len({c1_idx, c2_idx, c3_idx}) < 3:
                constraint_penalty += 0.3 * flight_revenue
            if any(c in self.used_crew for c in [c1_idx, c2_idx, c3_idx]):
                constraint_penalty += 0.2 * flight_revenue

            # Calculate costs
            flight_cost = (Cj * Di * phi +
                           Sm_p[p1_idx] * Di + Sm_p[p2_idx] * Di +
                           Sn_c[c1_idx] * Di + Sn_c[c2_idx] * Di + Sn_c[c3_idx] * Di)

            flight_emissions = Ej * omega * (Di ** 1.5)

            # Update tracking
            self.total_revenue += flight_revenue
            self.total_cost += flight_cost
            self.total_emissions += flight_emissions
            self.used_pilots.update([p1_idx, p2_idx])
            self.used_crew.update([c1_idx, c2_idx, c3_idx])
            self.pilot_hours[p1_idx] += Di
            self.pilot_hours[p2_idx] += Di
            self.crew_hours[c1_idx] += Di
            self.crew_hours[c2_idx] += Di
            self.crew_hours[c3_idx] += Di

            # Immediate reward for this flight (normalize to reasonable scale)
            reward = (flight_revenue - flight_cost) / 1000.0 - constraint_penalty / 1000.0

            # Small penalties for resource usage to encourage efficiency
            pilot_usage_penalty = 0.01 * np.sum(np.maximum(0, self.pilot_hours - Hp) / Hp)
            crew_usage_penalty = 0.01 * np.sum(np.maximum(0, self.crew_hours - Hc)/ Hc)
            reward -= pilot_usage_penalty + crew_usage_penalty

        # Move to next flight
        self.current_flight += 1
        terminated = self.current_flight >= self.num_flights

        if terminated:
            # Final episode reward based on overall performance
            budget_penalty = max(0, (self.total_cost - self.budget_cap) / self.budget_cap) * 2.0
            emission_penalty = max(0, (self.total_emissions - self.emission_limit * self.num_flights) /
                                   (self.emission_limit * self.num_flights)) * 2.0

            final_reward = (self.total_revenue / 10000.0 -
                            budget_penalty - emission_penalty)
            reward += final_reward

            # Reset for next episode
            self.current_flight = 0

        # Update state representation
        self._update_state()

        truncated = False
        info = {
            "revenue": getattr(self, 'total_revenue', 0),
            "total_cost": getattr(self, 'total_cost', 0),
            "emissions": getattr(self, 'total_emissions', 0),
            "current_flight": self.current_flight,
        }

        return self.state, reward, terminated, truncated, info

    def _update_state(self):
        """Update state representation matching original structure"""
        # Create scheduled flags for all flights (0 for future flights, actual values for processed ones)
        scheduled_flags = np.zeros(self.num_flights)
        if hasattr(self, 'processed_flights'):
            for i in self.processed_flights:
                scheduled_flags[i] = 1

        # Get flight data
        Ri = self.data["flight_revenue"]
        Pi = self.data["flight_priority"]
        Di = self.data["flight_duration"]
        phi = self.data["weather_based_fuel_degradation_factor"]
        omega = self.data["weather_based_emission_amplification_factor"]

        # Flight features (matching original: 6 features * num_flights)
        flight_features = np.stack([
            scheduled_flags,
            Ri,
            Pi,
            Di,
            phi,
            omega
        ], axis=1).flatten()  # Shape: (num_flights * 6,)

        # Resource states
        Hp = self.data["max_hours_pilot"]
        Hc = self.data["max_hours_crew"]

        pilot_hour_state = np.clip((Hp - getattr(self, 'pilot_hours', np.array(self.data["logged_hours_pilot"]))) / Hp,
                                   0, 1)
        crew_hour_state = np.clip((Hc - getattr(self, 'crew_hours', np.array(self.data["logged_hours_crew"]))) / Hc, 0,
                                  1)

        budget_state = np.array([(self.budget_cap - getattr(self, 'total_cost', 0)) / self.budget_cap])
        emission_state = np.array([getattr(self, 'total_emissions', 0) / (self.emission_limit * self.num_flights)])

        # Combine all state components
        self.state = np.concatenate([
            flight_features,  # num_flights * 6 dimensions
            budget_state,  # 1 dimension
            pilot_hour_state,  # num_pilots dimensions
            crew_hour_state,  # num_crew dimensions
            emission_state  # 1 dimension
        ]).astype(np.float32)

        # Debug: Print actual shape to verify
        print(f"State shape: {self.state.shape}, Expected: (612,)")
        if self.state.shape[0] != 612:
            print(f"Flight features: {flight_features.shape}")
            print(f"Budget state: {budget_state.shape}")
            print(f"Pilot hour state: {pilot_hour_state.shape}")
            print(f"Crew hour state: {crew_hour_state.shape}")
            print(f"Emission state: {emission_state.shape}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs_dim = (
            self.num_flights * 6 + 1 + self.num_pilots + self.num_crew + 1
        )
        self.state = np.zeros(obs_dim, dtype=np.float32)
        return self.state, {}