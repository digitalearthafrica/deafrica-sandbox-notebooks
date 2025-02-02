import pandas as pd

# Function to merge daily_volumes and DEA water levels datasets
def merge_datasets(daily_volumes_csv, dea_water_levels_csv, output_csv):
    # Read the daily volumes CSV
    daily_volumes_df = pd.read_csv(daily_volumes_csv)

    # Read the DEA water levels CSV
    dea_water_levels_df = pd.read_csv(dea_water_levels_csv)

    # Print column names to help debug the issue
    print("DEA Water Levels CSV Columns:", dea_water_levels_df.columns)

    # Check if 'Date' column exists, otherwise try other alternatives
    if 'Date' not in dea_water_levels_df.columns:
        if 'date' in dea_water_levels_df.columns:
            dea_water_levels_df['Date'] = dea_water_levels_df['date']
        elif 'DATE' in dea_water_levels_df.columns:
            dea_water_levels_df['Date'] = dea_water_levels_df['DATE']
        else:
            raise KeyError("No 'Date' column found in DEA dataset. Please check the file structure.")

    # Ensure the 'Date' column is in datetime format in both dataframes
    daily_volumes_df['Date'] = pd.to_datetime(daily_volumes_df['Date'], format="%Y-%m-%d", errors='coerce')
    dea_water_levels_df['Date'] = pd.to_datetime(dea_water_levels_df['Date'], format="%Y-%m-%d", errors='coerce')

    # Merge the datasets on the 'Date' column
    merged_df = pd.merge(daily_volumes_df, dea_water_levels_df, on='Date', how='inner')

    # Select the required columns for the final dataset
    merged_df = merged_df[['Date', 'COR_LEVEL', 'Dam_Level', 'Volume_mcm', 'water_area_ha', 'percent_invalid']]

    # Save the merged dataset to a new CSV file
    merged_df.to_csv(output_csv, index=False)
    print(f"Merged data saved to {output_csv}")

# Example usage
daily_volumes_csv = "DWS/daily_volumes.csv"
dea_water_levels_csv = "DEA/dea_water_levels_raw.csv"
output_csv = "training_data.csv"

merge_datasets(daily_volumes_csv, dea_water_levels_csv, output_csv)
