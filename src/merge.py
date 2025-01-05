import os
import json
import pandas as pd
from datetime import datetime

CSV_PATH = "data/air_quality/london-air-quality-normalized.csv"

JSON_FOLDER = "data/split_json"

OUTPUT_FOLDER = "data/merged_json"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

df = pd.read_csv(CSV_PATH)

pollution_by_date = {}
for _, row in df.iterrows():
    dt = datetime.strptime(row["date"], "%Y/%m/%d")
    date_str = dt.strftime("%Y-%m-%d")

    pollution_by_date[date_str] = {
        "pm25": row["pm25"],
        "pm10": row["pm10"],
        "o3":   row["o3"],
        "no2":  row["no2"],
        "so2":  row["so2"],
        "co":   row["co"]
    }

for date_str, pollution_data in pollution_by_date.items():
    json_filename = f"{date_str}.json"
    json_path = os.path.join(JSON_FOLDER, json_filename)
    
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        data["pollution"] = pollution_data
        output_path = os.path.join(OUTPUT_FOLDER, json_filename)
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(data, out_f, ensure_ascii=False, indent=2)

        print(f"Saved merged JSON for {date_str} -> {output_path}")
    else:
        print(f"No JSON file found for {date_str}, skipping...")
