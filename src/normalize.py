import csv
import os

def normalize_air_quality_data(
    input_path='data/air_quality/london-air-quality.csv',
    output_path='data/air_quality/london-air-quality-normalized.csv'
):
    with open(input_path, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        cleaned_rows = []
        
        for row in reader:
            if (row['pm25'].strip() == '' or
                row['pm10'].strip() == '' or
                row['o3'].strip() == '' or
                row['no2'].strip() == ''):
                continue  
            
            if row['so2'].strip() == '':
                row['so2'] = '0'
            if row['co'].strip() == '':
                row['co'] = '0'
            
            cleaned_rows.append(row)
    
    with open(output_path, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)

def main():
    input_csv = 'data/air_quality/london-air-quality.csv'
    output_csv = 'data/air_quality/london-air-quality-normalized.csv'

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    normalize_air_quality_data(input_csv, output_csv)
    print(f'Normalized air quality data has been saved to {output_csv}')

if __name__ == '__main__':
    main()
