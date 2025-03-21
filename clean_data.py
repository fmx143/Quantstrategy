'''
Cleaning the CSV if it does not contains all required columns.
'''

import pandas as pd
from apy import *


# Path to the original CSV file (inside apy.py file)
data_csv_path = eu_15min

# Function to check for the minimum required columns in CSV file for data
def check_required_columns(data):
    required_columns = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
    columns_present = all(column in data.columns for column in required_columns)
    print(columns_present)

# Function to clean the CSV data file
def clean_csv_data(file_path, cleaned_file_path):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
    if check_required_columns(df):
        # Separate the combined datetime column into separate date and time columns
        df[['Date', 'Time']] = df['DateTime'].str.split(' ', expand=True)
        df.drop(columns=['DateTime'], inplace=True)
       # Reorder the columns to have Date and Time in the first place
        df = df[['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']]
        # Save the cleaned data to a new CSV file
        df.to_csv(cleaned_file_path, index=False)
        print("Cleaned data saved to:", cleaned_file_path)
        cleaned_df = pd.read_csv(cleaned_file_path)
        print(cleaned_df.head())
        return cleaned_df
    else:
        print("The CSV file does not contain all required columns.")
        return None

# Path to the cleaned CSV file
cleaned_csv_path = 'ForexSB_EURUSD_M15_06.02.2017-14.02.2025_cleaned.csv'

# Call the function and clean the CSV data
cleaned_df = clean_csv_data(data_csv_path, cleaned_csv_path)