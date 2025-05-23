import pandas as pd
import json


def prep_data(filenames, jsonfile):
    master_dict = {}
    for file in filenames:
        df = pd.read_csv(file)
        for col in df.columns:
            if col in master_dict:
                raise ValueError(f"Duplicate column name '{col}' found across CSV files.")
            master_dict[col] = df[col].to_numpy()

        # Load and add JSON
        with open(jsonfile, "r") as f:
            json_data = json.load(f)
            master_dict.update(json_data)

    return master_dict