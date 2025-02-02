import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Function to process each line of data
def process_line(line):
    data_line = ' '.join(line.split())
    data_pieces = data_line.split(' ')
    
    if len(data_pieces) >= 6:
        date_str = data_pieces[0]
        time_str = data_pieces[1]
        
        # Parse date and time
        try:
            date = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
            time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
        except ValueError:
            return None
        
        # Extract remaining fields
        corrected_level = data_pieces[2]
        level_qua = data_pieces[3]
        corrected_flow = data_pieces[4]
        flow_qua = data_pieces[5]
        
        return {
            'Date': date,
            'Time': time,
            'COR_LEVEL': corrected_level,
            'QUA_LEVEL': level_qua,
            'COR_FLOW': corrected_flow,
            'QUA_FLOW': flow_qua
        }
    return None

# Retry configuration for HTTP requests
def requests_retry_session(
    retries=5,
    backoff_factor=1,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# Function to download and process data
def download_and_process_data(station_id, start_date, end_date, base_url, output_csv):
    processed_data = []
    current_start_date = start_date

    while current_start_date <= end_date:
        query_url = f"{base_url}?Station={station_id}&DataType=Point&StartDT={current_start_date}&EndDT={end_date}&SiteType=RES"
        print(f"Downloading data from {current_start_date} to {end_date}...")

        try:
            response = requests_retry_session().get(query_url, timeout=10)
            response.raise_for_status()  # Raise an error for bad responses
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            break

        lines = response.text.splitlines()
        for line in lines:
            processed = process_line(line)
            if processed:
                processed_data.append(processed)

        df = pd.DataFrame(processed_data, columns=['Date', 'Time', 'COR_LEVEL', 'QUA_LEVEL', 'COR_FLOW', 'QUA_FLOW'])
        df.to_csv(output_csv, mode='a', header=not bool(processed_data), index=False)

        current_start_date = (datetime.strptime(processed_data[-1]['Date'], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

        time.sleep(1)

    print(f"Data successfully saved to {output_csv}")

# Example usage
base_url = "https://www.dws.gov.za/Hydrology/Verified/HyData.aspx"
station_id = "B3R002100.00"
start_date = "2000-01-01"
end_date = "2024-03-05"
output_csv = "processed_hydrology_data.csv"

download_and_process_data(station_id, start_date, end_date, base_url, output_csv)
