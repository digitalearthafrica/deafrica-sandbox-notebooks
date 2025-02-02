import pandas as pd
from tqdm import tqdm

# Function to add volume calculations to daily hydrology data
def add_volume_to_daily_data(input_csv, rating_curve_csv, output_csv):
    # Read the daily hydrology CSV file
    df = pd.read_csv(input_csv)

    # Ensure that the 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d", errors='coerce')

    # Convert COR_LEVEL and COR_FLOW to numeric, ignoring errors
    df['COR_LEVEL'] = pd.to_numeric(df['COR_LEVEL'], errors='coerce')
    df['COR_FLOW'] = pd.to_numeric(df['COR_FLOW'], errors='coerce')

    # Read the rating curve CSV (water_level, volume_mcm)
    rating_curve_df = pd.read_csv(rating_curve_csv)

    # Ensure the rating curve columns are numeric
    rating_curve_df['water_level'] = pd.to_numeric(rating_curve_df['water_level'], errors='coerce')
    rating_curve_df['volume_mcm'] = pd.to_numeric(rating_curve_df['volume_mcm'], errors='coerce')

    # Add the spill level to convert it to dam level
    df['Dam_Level'] = df['COR_LEVEL'] + 28.08

    # Function to find the closest volume for a given dam level
    def find_closest_volume(dam_level):
        idx = (rating_curve_df['water_level'] - dam_level).abs().idxmin()
        return rating_curve_df.loc[idx, 'volume_mcm']

    # Calculate the volume for each row based on dam level
    df['Volume_mcm'] = df['Dam_Level'].apply(find_closest_volume)

    # Save the daily data with volumes to a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Daily data with volumes saved to {output_csv}")

# Example usage
input_csv = "daily_hydrology_data.csv"
rating_curve_csv = "rating_curve.csv"
output_csv = "daily_volumes.csv"

add_volume_to_daily_data(input_csv, rating_curve_csv, output_csv)
