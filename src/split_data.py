import os
import json
from datetime import datetime
from collections import defaultdict

def split_weather_json_by_day(input_filepath, output_dir):
    with open(input_filepath, 'r') as f:
        weather_data = json.load(f)

    valid_times_str = weather_data["coords"]["valid_time"]["data"]
    valid_times_dt = [datetime.fromisoformat(ts) for ts in valid_times_str]

    day_to_indices = defaultdict(list)
    for idx, dt_obj in enumerate(valid_times_dt):
        day_str = dt_obj.strftime("%Y-%m-%d")
        day_to_indices[day_str].append(idx)

    data_vars = weather_data.get("data_vars", {})
    coords = weather_data.get("coords", {})

    for day_str, indices in day_to_indices.items():
        indices_sorted = sorted(indices)

        new_json = {
            "coords": {},
            "data_vars": {}
        }

        for coord_key, coord_val in coords.items():
            data_field = coord_val.get("data", None)
            if data_field is None:
                continue

            if isinstance(data_field, list) and len(data_field) == len(valid_times_str):
                new_data = [data_field[i] for i in indices_sorted]
            else:
                new_data = data_field

            new_json["coords"][coord_key] = {
                "data": new_data
            }

        for var_name, var_info in data_vars.items():
            var_data = var_info.get("data", None)
            if var_data is None:
                continue

            if len(var_data) == len(valid_times_str):
                sliced_data = [var_data[i] for i in indices_sorted]
            else:
                sliced_data = var_data

            new_json["data_vars"][var_name] = {
                "data": sliced_data
            }

        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{day_str}.json"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, 'w') as out_f:
            json.dump(new_json, out_f, indent=2)

def main():
    input_folder = "data/weather_json"
    output_folder = "data/split_json"

    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            in_file = os.path.join(input_folder, filename)
            split_weather_json_by_day(in_file, output_folder)

    print("Done splitting JSON files by day ")

if __name__ == "__main__":
    main()
