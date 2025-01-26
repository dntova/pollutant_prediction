import requests
import pandas as pd
import os
from datetime import datetime, timedelta
from tqdm import tqdm
import time

# Define the output directory
waqi_output_dir = 'data/waqi_air_quality'
os.makedirs(waqi_output_dir, exist_ok=True)

# Define the base URL
base_url = 'https://api.waqi.info/feed/athens/'

# Your WAQI API token
api_token = 'f55a05f44b21117e6aea16d6dc0ce899747fdd6f'

# Define date range
start_date = datetime(2024, 6, 30)
end_date = datetime(2024, 11, 30)

# Define pollutants
pollutants = ['pm25', 'pm10', 'no2', 'o3', 'so2', 'co']

# Define batch size (e.g., daily)
batch_size = timedelta(days=1)

current_date = start_date
while current_date <= end_date:
    date_str = current_date.strftime('%Y-%m-%d')
    params = {
        'token': api_token,
        'date': date_str
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'ok':
            # Assuming WAQI provides historical data in the response
            # Adjust based on actual API response structure
            measurements = data['data']['iaqi']
            # Convert to DataFrame
            df = pd.DataFrame([measurements])
            df['date'] = date_str
            # Save to CSV
            output_filename = f'waqi_air_quality_{date_str}.csv'
            output_path = os.path.join(waqi_output_dir, output_filename)
            df.to_csv(output_path, index=False)
            print(f"Downloaded WAQI data for {date_str}")
        else:
            print(f"No WAQI data available for {date_str}")
    else:
        print(f"Failed to retrieve WAQI data for {date_str}: {response.status_code}")
    time.sleep(1)  # Respect API rate limits
    current_date += batch_size
 