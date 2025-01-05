#!/usr/bin/env python3

import os
import json
import xarray as xr
import datetime

BASE_DIR = "data/weather"
OUTPUT_DIR = "data/weather_json"

def default_converter(obj):
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    
    raise TypeError(f"Type {type(obj)} not serializable")

def convert_era5_nc_to_json():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for day_dir in os.listdir(BASE_DIR):
        full_day_path = os.path.join(BASE_DIR, day_dir)

        if not os.path.isdir(full_day_path):
            continue
        if not day_dir.startswith("era5_weather_"):
            continue

        accum_file = os.path.join(full_day_path, "data_stream-oper_stepType-accum.nc")
        instant_file = os.path.join(full_day_path, "data_stream-oper_stepType-instant.nc")

        if not (os.path.isfile(accum_file) and os.path.isfile(instant_file)):
            print(f"Skipping {day_dir}: missing accum/instant files.")
            continue

        print(f"Processing {day_dir}...")

        ds_accum = xr.open_dataset(accum_file)
        ds_instant = xr.open_dataset(instant_file)
        ds_merged = xr.merge([ds_accum, ds_instant])

        data_dict = ds_merged.to_dict(data=True)

        output_filename = f"{day_dir}.json"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        with open(output_path, "w") as f:
            json.dump(data_dict, f, indent=2, default=default_converter)

        ds_accum.close()
        ds_instant.close()
        print(f"Finished writing {output_path}.")

if __name__ == "__main__":
    convert_era5_nc_to_json()
