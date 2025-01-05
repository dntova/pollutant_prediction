import pandas as pd

def create_daily_averages(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    
    df['time'] = pd.to_datetime(df['time'])
    
    df['date'] = df['time'].dt.date
    
    daily_averages = df.groupby('date', as_index=False).mean()

    if 'time' in daily_averages.columns:
        daily_averages.drop(columns=['time'], inplace=True)

    daily_averages = daily_averages.round(8)
    
    daily_averages.to_csv(output_csv, index=False)

if __name__ == "__main__":
    input_file = "data/all_weather_pollution.csv"
    output_file = "data/daily_averages.csv"
    
    create_daily_averages(input_file, output_file)
    print(f"Daily averages saved to {output_file}.")
