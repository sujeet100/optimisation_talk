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
        self.action_space = spaces.MultiDiscrete(
            [2, self.num_aircraft, self.num_pilots, self.num_pilots, self.num_crew, self.num_crew, self.num_crew] * self.num_flights
        )

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
        alpha = 0.5
        beta = 0.5
        a = 0.3
        b = 1.5

        action = np.array(action).reshape(self.num_flights, 7)

        Ri, Pi, Di = self.data["flight_revenue"], self.data["flight_priority"], self.data["flight_duration"]
        Cj, Ej, Fj = self.data["op_cost_per_km"], self.data["carbon_emission_gm_per_km"], self.data["fuel_efficiency_km_per_kg"]
        Sm_p, Hm_p = self.data["salary_pilot"], self.data["logged_hours_pilot"]
        Sn_c, Hn_c = self.data["salary_crew"], self.data["logged_hours_crew"]
        phi, omega = self.data["weather_based_fuel_degradation_factor"], self.data["weather_based_emission_amplification_factor"]
        Hp, Hc = self.data["max_hours_pilot"], self.data["max_hours_crew"]

        total_cost = 0
        total_emissions = 0
        revenue = 0
        pilot_hours = np.array(Hm_p)
        crew_hours = np.array(Hn_c)
        scheduled_flags = np.zeros(self.num_flights)
        used_pilots = set()
        used_crew = set()

        for i, (w_i, a_idx, p1_idx, p2_idx, c1_idx, c2_idx, c3_idx) in enumerate(action):
            if w_i == 1:
                # Hard constraints: 2 distinct pilots, 3 distinct crew
                if len({p1_idx, p2_idx}) < 2 or len({c1_idx, c2_idx, c3_idx}) < 3:
                    continue

                # No reuse of pilots or crew
                if any(p in used_pilots for p in [p1_idx, p2_idx]) or any(c in used_crew for c in [c1_idx, c2_idx, c3_idx]):
                    continue

                used_pilots.update([p1_idx, p2_idx])
                used_crew.update([c1_idx, c2_idx, c3_idx])
                scheduled_flags[i] = 1
                revenue += Ri[i] * (1 + a * Pi[i])
                total_cost += Cj[a_idx] * Di[i] * phi[i]
                total_cost += Sm_p[p1_idx] * Di[i] + Sm_p[p2_idx] * Di[i]
                total_cost += Sn_c[c1_idx] * Di[i] + Sn_c[c2_idx] * Di[i] + Sn_c[c3_idx] * Di[i]
                pilot_hours[p1_idx] += Di[i]
                pilot_hours[p2_idx] += Di[i]
                crew_hours[c1_idx] += Di[i]
                crew_hours[c2_idx] += Di[i]
                crew_hours[c3_idx] += Di[i]
                total_emissions += Ej[a_idx] * omega[i] * (Di[i] ** b)

        budget_overrun = max(0, total_cost - self.budget_cap)
        penalty_budget = ((alpha + 1) * (budget_overrun)) / (alpha * budget_overrun + 1)

        emission_violation = max(0, total_emissions - self.emission_limit * self.num_flights)
        penalty_emission = 1e3 if emission_violation > 0 else 0

        pilot_hours_over = np.maximum(0, pilot_hours - Hp)
        crew_hours_over = np.maximum(0, crew_hours - Hc)
        largest_pilot_hours_over = np.max(pilot_hours_over)
        largest_crew_hours_over = np.max(crew_hours_over)
        penalty_pilot = ((beta + 1) * largest_pilot_hours_over) / (beta * largest_pilot_hours_over + 1)
        penalty_crew = ((beta + 1) * largest_crew_hours_over) / (beta * largest_crew_hours_over + 1)

        reward = revenue - (penalty_budget + penalty_emission + penalty_pilot + penalty_crew)

        flight_features = np.stack([
            scheduled_flags,
            Ri,
            Pi,
            Di,
            phi,
            omega
        ], axis=1).flatten()

        pilot_hour_state = np.clip((Hp - pilot_hours) / Hp, 0, 1)
        crew_hour_state = np.clip((Hc - crew_hours) / Hc, 0, 1)
        budget_state = np.array([(self.budget_cap - total_cost) / self.budget_cap])
        emission_state = np.array([total_emissions / (self.emission_limit * self.num_flights)])

        self.state = np.concatenate([
            flight_features,
            budget_state,
            pilot_hour_state,
            crew_hour_state,
            emission_state
        ]).astype(np.float32)

        terminated = True
        truncated = False
        info = {
            "revenue": revenue,
            "total_cost": total_cost,
            "budget_overrun": budget_overrun,
            "emissions": total_emissions,
            "emission_violation": emission_violation,
            "pilot_hours_over": pilot_hours_over.tolist(),
            "crew_hours_over": crew_hours_over.tolist()
        }
        return self.state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs_dim = (
            self.num_flights * 6 + 1 + self.num_pilots + self.num_crew + 1
        )
        self.state = np.zeros(obs_dim, dtype=np.float32)
        return self.state, {}