import pandas as pd
from tqdm import tqdm

# Function to remove duplicates and convert sub-daily data to daily averages with progress
def convert_to_daily_averages(input_csv, output_csv):
    # Read the CSV file without headers
    df = pd.read_csv(input_csv, header=None)

    # Manually assign column names (since the original file has no headers)
    df.columns = ['Date', 'Time', 'COR_LEVEL', 'QUA_LEVEL', 'COR_FLOW', 'QUA_FLOW']

    # Ensure that the 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d", errors='coerce')

    # Convert the COR_LEVEL and COR_FLOW columns to numeric (ignoring errors)
    df['COR_LEVEL'] = pd.to_numeric(df['COR_LEVEL'], errors='coerce')
    df['COR_FLOW'] = pd.to_numeric(df['COR_FLOW'], errors='coerce')

    # Drop duplicate rows based on 'Date', 'COR_LEVEL', and 'COR_FLOW' columns
    df = df.drop_duplicates(subset=['Date', 'COR_LEVEL', 'COR_FLOW'])

    # Show progress while grouping the data by date
    daily_data = []
    unique_dates = df['Date'].dropna().unique()  # Drop NaN dates

    # Using tqdm to track progress
    for date in tqdm(unique_dates, desc="Processing daily data"):
        day_data = df[df['Date'] == date]
        avg_cor_level = day_data['COR_LEVEL'].mean()
        avg_cor_flow = day_data['COR_FLOW'].mean()
        daily_data.append([date, avg_cor_level, avg_cor_flow])

    # Convert the list to a DataFrame
    daily_df = pd.DataFrame(daily_data, columns=['Date', 'COR_LEVEL', 'COR_FLOW'])

    # Save the daily averages to a new CSV file
    daily_df.to_csv(output_csv, index=False)
    print(f"Daily averages saved to {output_csv}")

# Example usage
input_csv = "processed_hydrology_data.csv"
output_csv = "daily_hydrology_data.csv"

convert_to_daily_averages(input_csv, output_csv)
