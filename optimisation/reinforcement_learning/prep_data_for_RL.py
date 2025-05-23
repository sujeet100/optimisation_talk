import pandas as pd
import json


def prep_data():
    filenames =["./data_sim/aircraft_data.csv", "./data_sim/pilot_data.csv", "./data_sim/crew_data.csv", "./data_sim/flight_data.csv"]
    master_dict = {}
    for file in filenames:
        df = pd.read_csv(file)
        for col in df.columns:
            if col in master_dict:
                raise ValueError(f"Duplicate column name '{col}' found across CSV files.")
            master_dict[col] = df[col].to_numpy()

        # Load and add JSON
        with open("./data_sim/opt_params.json", "r") as f:
            json_data = json.load(f)
            master_dict.update(json_data)

    return master_dict

# if __name__ == "__main__":
#     data_for_RL = prep_data()
#     print("something")