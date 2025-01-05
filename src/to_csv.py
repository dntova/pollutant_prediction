import os
import glob
import json
import numpy as np
import pandas as pd

def parse_weather_json(file_path, agg_func=np.mean):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    times = data["coords"]["valid_time"]["data"]
    
    pm25 = data["pollution"].get("pm25", None)
    pm10 = data["pollution"].get("pm10", None)
    o3   = data["pollution"].get("o3", None)
    no2  = data["pollution"].get("no2", None)
    so2  = data["pollution"].get("so2", None)
    co   = data["pollution"].get("co", None)

    def get_variable(var_name):
        return np.array(data["data_vars"][var_name]["data"], dtype=np.float32)
    
    tp_3d  = get_variable("tp")
    t2m_3d = get_variable("t2m")
    u10_3d = get_variable("u10")
    v10_3d = get_variable("v10")
    sp_3d  = get_variable("sp")
    
    records = []
    for t_idx, time_val in enumerate(times):
        tp_val  = agg_func(tp_3d[t_idx])
        t2m_val = agg_func(t2m_3d[t_idx])
        u10_val = agg_func(u10_3d[t_idx])
        v10_val = agg_func(v10_3d[t_idx])
        sp_val  = agg_func(sp_3d[t_idx])
        
        record = {
            "time": time_val,
            "tp":   tp_val,
            "t2m":  t2m_val,
            "u10":  u10_val,
            "v10":  v10_val,
            "sp":   sp_val,
            "pm25": pm25,
            "pm10": pm10,
            "o3":   o3,
            "no2":  no2,
            "so2":  so2,
            "co":   co
        }
        records.append(record)
    
    return records


def create_csv_from_json(folder_path, output_csv_path, agg_func=np.mean):
    """
    Reads all JSON files from folder_path, extracts aggregated weather+pollution data,
    compiles into a single CSV (omitting lat/lon).
    Also shows an example correlation analysis at the end.
    """
    all_records = []
    
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    for jf in json_files:
        records = parse_weather_json(jf, agg_func=agg_func)
        all_records.extend(records)
    
    df = pd.DataFrame(all_records)
    
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    
    df.sort_values(by='time', inplace=True)
    
    df.to_csv(output_csv_path, index=False)
    print(f"CSV created: {output_csv_path}")
    


if __name__ == "__main__":
    input_folder = "data/merged_json"
    output_csv = "data/all_weather_pollution.csv"
    
    create_csv_from_json(input_folder, output_csv, agg_func=np.mean)
