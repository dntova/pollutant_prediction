import os
import subprocess

folder = "data/weather"

for file_name in os.listdir(folder):
    if not file_name.endswith(".nc"):
        continue

    nc_path = os.path.join(folder, file_name)
    zip_name = file_name[:-3] + ".zip"
    zip_path = os.path.join(folder, zip_name)
    os.rename(nc_path, zip_path)

    output_folder = os.path.splitext(zip_path)[0]
    os.makedirs(output_folder, exist_ok=True)
    subprocess.run(["unzip", zip_path, "-d", output_folder])

    os.remove(zip_path)
