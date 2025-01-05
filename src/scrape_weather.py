import cdsapi
import pandas as pd
import os
import time
from tqdm import tqdm

c = cdsapi.Client()

weather_output_dir = 'data/weather'
os.makedirs(weather_output_dir, exist_ok=True)

start_date = '2024-01-01'
end_date = '2024-11-30'

dates = pd.date_range(start=start_date, end=end_date, freq='D')

batch_size = 7

# Define the geographic area for London
london_area = [
    51.70, -0.5, 51.30, 0.3
]

def chunk_dates(date_list, batch_size):
    for i in range(0, len(date_list), batch_size):
        yield date_list[i:i + batch_size]

def format_dates(batch):
    years = sorted(list(set([date.strftime('%Y') for date in batch])))
    months = sorted(list(set([date.strftime('%m') for date in batch])))
    days = sorted(list(set([date.strftime('%d') for date in batch])))
    return years, months, days

def download_weather(batch, batch_output_filename):
    years, months, days = format_dates(batch)
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    '2m_temperature',
                    'total_precipitation',
                    '10m_u_component_of_wind',
                    '10m_v_component_of_wind',
                    'surface_pressure',
                    'weather_symbol_1h',
                ],
                'year': years,
                'month': months,
                'day': days,
                'time': [
                    '08:00', '12:00', '18:00',
                ],
                'area': london_area,
            },
            os.path.join(weather_output_dir, batch_output_filename))
        print(f"Weather data downloaded for dates: {batch[0].strftime('%Y-%m-%d')} to {batch[-1].strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"Failed to download weather data for dates: {batch[0].strftime('%Y-%m-%d')} to {batch[-1].strftime('%Y-%m-%d')}")
        print(f"Error: {e}")


for batch in tqdm(list(chunk_dates(dates, batch_size)), desc="Downloading Data"):
    batch_start = batch[0].strftime('%Y%m%d')
    batch_end = batch[-1].strftime('%Y%m%d')
    
    weather_filename = f'era5_weather_{batch_start}.nc'
    
    download_weather(batch, weather_filename)
    
    time.sleep(1)

print("Data download completed.")
