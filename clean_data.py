import pandas as pd
import os
import warnings
from apy import *  # Assuming apy.py defines eu_15min as the CSV file path

# Path to the original CSV file (in your cloud)
original_csv_path = eu_15min

def check_required_columns(data):
    """
    Checks if the CSV data has the required columns.
    If any required columns are missing, a warning is issued.
    """
    required_columns = ['DateTime', 'Open', 'High', 'Low', 'Close']
    missing = [col for col in required_columns if col not in data.columns]
    
    if missing:
        warnings.warn(f"WARNING: The CSV file is missing the following required columns: {missing}")
        return False
    else:
        print("All required columns are present.")
        return True

def clean_csv_data(file_path):
    """
    Reads the CSV file, verifies that required columns are present,
    processes the DateTime column, and saves the cleaned data to a new file.
    The new file is saved in the current working directory (e.g., your VS Code workspace)
    with '_cleand' inserted before the .csv extension.
    """
    try:
        # Read CSV using tab as the separator, with no header and providing column names
        df = pd.read_csv(file_path, sep='\t', header=None, 
                         names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
        print("Preview of original data:")
        print(df.head())

        if not check_required_columns(df):
            print("CSV file does not contain all required columns. Aborting cleaning.")
            return None

        # Process the DateTime column
        try:
            # Try converting to datetime directly
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            print("DateTime column is in the correct format.")
            df.rename(columns={'DateTime': 'datetime'}, inplace=True)
        except Exception:
            # Attempt to fix if DateTime is not directly convertible:
            try:
                df[['Date', 'Time']] = df['DateTime'].str.split(' ', expand=True)
                df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
                df.drop(columns=['Date', 'Time', 'DateTime'], inplace=True)
                print("Combined Date and Time columns into a single datetime column.")
            except Exception as e:
                warnings.warn(f"Error processing DateTime column: {e}")
                return None

        # Reorder columns so that 'datetime' comes first
        required_order = ['datetime', 'Open', 'High', 'Low', 'Close']
        extra_columns = [col for col in df.columns if col not in required_order]
        df = df[required_order + extra_columns]

        # Create a new file name using the original file's base name, but saving locally
        original_filename = os.path.basename(file_path)  # e.g., ForexSB_EURUSD_M15_06.02.2017-14.02.2025.csv
        base, ext = os.path.splitext(original_filename)
        new_filename = base + '_cleaned' + ext
        # Save to current working directory (VS Code local directory)
        new_file_path = os.path.join(os.getcwd(), new_filename)
        
        # Save the cleaned DataFrame to the new file
        df.to_csv(new_file_path, index=False)
        print("Cleaned CSV saved as:", new_file_path)
        return df
    except Exception as e:
        warnings.warn(f"An error occurred while processing the file: {e}")
        return None

# Call the function and process the CSV data
cleaned_df = clean_csv_data(original_csv_path)

if cleaned_df is None:
    raise ValueError("Failed to clean the CSV data. Please check the input file and try again.")

