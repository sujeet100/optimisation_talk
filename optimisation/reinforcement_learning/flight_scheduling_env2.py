import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FlightSchedulingEnv(gym.Env):
    def __init__(self, data):
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
            self.num_aircraft,
            self.num_pilots,
            self.num_pilots,
            self.num_crew,
            self.num_crew,
            self.num_crew
        ])

        # Observation space
        obs_dim = (
            self.num_flights * 6 + 1 + self.num_pilots + self.num_crew + 1 + self.num_pilots + self.num_crew + self.num_aircraft
        )
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.state = np.zeros(obs_dim, dtype=np.float32)
        self.reset()

    def step(self, action):
        if not hasattr(self, 'current_flight'):
            self.current_flight = 0
            self.total_revenue = 0
            self.total_cost = 0
            self.total_emissions = 0
            self.pilot_hours = np.array(self.data["logged_hours_pilot"]).copy()
            self.crew_hours = np.array(self.data["logged_hours_crew"]).copy()
            self.used_pilots = set()
            self.used_crew = set()
            self.used_aircraft = set()
            self.aircraft_load = np.zeros(self.num_aircraft)
            self.scheduled_flights = np.zeros(self.num_flights)

        flight_idx = self.current_flight
        action = np.asarray(action).flatten()
        assert action.shape == (7,), f"Invalid action shape: {action.shape}"
        w_i, a_idx, p1_idx, p2_idx, c1_idx, c2_idx, c3_idx = map(int, action)

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

        if w_i == 1:
            self.scheduled_flights[flight_idx] = 1
            flight_revenue = Ri * (1 + 0.3 * Pi)
            constraint_penalty = 0

            # Hard assignment constraint violations
            unique_pilots = len({p1_idx, p2_idx})
            unique_crew = len({c1_idx, c2_idx, c3_idx})

            if unique_pilots < 2 or unique_crew < 3:
                reward -= 2.0  # hard penalty
            if p1_idx in self.used_pilots or p2_idx in self.used_pilots:
                reward -= 2.0
            if any(c in self.used_crew for c in [c1_idx, c2_idx, c3_idx]):
                reward -= 2.0

            flight_cost = (Cj * Di * phi +
                           Sm_p[p1_idx] * Di + Sm_p[p2_idx] * Di +
                           Sn_c[c1_idx] * Di + Sn_c[c2_idx] * Di + Sn_c[c3_idx] * Di)
            flight_emissions = Ej * omega * (Di ** 1.5)

            self.total_revenue += flight_revenue
            self.total_cost += flight_cost
            self.total_emissions += flight_emissions
            self.used_pilots.update([p1_idx, p2_idx])
            self.used_crew.update([c1_idx, c2_idx, c3_idx])
            self.used_aircraft.add(a_idx)

            self.pilot_hours[p1_idx] += Di
            self.pilot_hours[p2_idx] += Di
            self.crew_hours[c1_idx] += Di
            self.crew_hours[c2_idx] += Di
            self.crew_hours[c3_idx] += Di
            self.aircraft_load[a_idx] += Di

            pilot_usage_penalty = 0.01 * np.sum(np.maximum(0, self.pilot_hours - Hp) / Hp)
            crew_usage_penalty = 0.01 * np.sum(np.maximum(0, self.crew_hours - Hc) / Hc)

            reward += (flight_revenue - flight_cost) / 1000.0 - pilot_usage_penalty - crew_usage_penalty

        self.current_flight += 1
        terminated = self.current_flight >= self.num_flights

        if terminated:
            budget_penalty = max(0, (self.total_cost - self.budget_cap) / self.budget_cap) * 2.0
            emission_penalty = max(0, (self.total_emissions - self.emission_limit * self.num_flights) /
                                   (self.emission_limit * self.num_flights)) * 2.0

            diversity_bonus = (
                len(self.used_pilots) / self.num_pilots +
                len(self.used_crew) / self.num_crew +
                len(self.used_aircraft) / self.num_aircraft
            )

            final_reward = (self.total_revenue / 10000.0 - budget_penalty - emission_penalty + diversity_bonus)
            reward += final_reward
            self.current_flight = 0

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
        scheduled_flags = np.zeros(self.num_flights)
        Ri = self.data["flight_revenue"]
        Pi = self.data["flight_priority"]
        Di = self.data["flight_duration"]
        phi = self.data["weather_based_fuel_degradation_factor"]
        omega = self.data["weather_based_emission_amplification_factor"]

        flight_features = np.stack([
            scheduled_flags,
            Ri,
            Pi,
            Di,
            phi,
            omega
        ], axis=1).flatten()

        Hp = self.data["max_hours_pilot"]
        Hc = self.data["max_hours_crew"]
        pilot_hour_state = np.clip((Hp - self.pilot_hours) / Hp, 0, 1)
        crew_hour_state = np.clip((Hc - self.crew_hours) / Hc, 0, 1)

        budget_state = np.array([(self.budget_cap - self.total_cost) / self.budget_cap])
        emission_state = np.array([self.total_emissions / (self.emission_limit * self.num_flights)])

        pilot_load = self.pilot_hours / (Hp * self.num_flights)
        crew_load = self.crew_hours / (Hc * self.num_flights)
        aircraft_load = self.aircraft_load / (np.max(self.aircraft_load) + 1e-6)

        self.state = np.concatenate([
            flight_features,
            budget_state,
            pilot_hour_state,
            crew_hour_state,
            emission_state,
            pilot_load,
            crew_load,
            aircraft_load
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_flight = 0
        self.total_revenue = 0
        self.total_cost = 0
        self.total_emissions = 0
        self.pilot_hours = np.array(self.data["logged_hours_pilot"]).copy()
        self.crew_hours = np.array(self.data["logged_hours_crew"]).copy()
        self.used_pilots = set()
        self.used_crew = set()
        self.used_aircraft = set()
        self.aircraft_load = np.zeros(self.num_aircraft)
        self.scheduled_flights = np.zeros(self.num_flights)
        self._update_state()
        return self.state, {}
