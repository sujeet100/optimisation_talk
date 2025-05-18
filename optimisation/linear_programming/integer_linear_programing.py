import pulp

# Create the LP problem (maximize profit)
model = pulp.LpProblem("Maximize_Aircraft_Profit", pulp.LpMaximize)

# Decision variables: number of each aircraft to use
xa = pulp.LpVariable('Airbus_A320', lowBound=0, upBound=100, cat='Integer')
xb = pulp.LpVariable('Boeing_737', lowBound=0, upBound=50, cat='Integer')

# Test data
cost_A320 = 500_000
cost_B737 = 400_000

crew_A320 = 6
crew_B737 = 4

profit_A320 = 50_000  # hypothetical per unit
profit_B737 = 45_000  # hypothetical per unit

budget_limit = 50_000_000  # e.g., 50 million dollars
crew_limit = 600  # total crew available

# Objective function: Maximize profit
model += profit_A320 * xa + profit_B737 * xb, "Total_Profit"

# Constraints
model += cost_A320 * xa + cost_B737 * xb <= budget_limit, "Budget_Constraint"
model += crew_A320 * xa + crew_B737 * xb <= crew_limit, "Crew_Constraint"

# Solve the model
model.solve()

# Output results
print("Status:", pulp.LpStatus[model.status])
print("Optimal number of Airbus A320 to use:", xa.varValue)
print("Optimal number of Boeing 737 to use:", xb.varValue)
print("Maximum Profit: $", pulp.value(model.objective))