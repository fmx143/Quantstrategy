'''
Cleaning the CSV if it does not contain all required columns.
'''

import pandas as pd
import os
from apy import *  # Assuming apy.py contains the path to the CSV file

# Path to the original CSV file (inside apy.py file)
# Change eu_15min to the path of the CSV file you want to clean
original_csv_path = eu_15min

# Function to check for the minimum required columns in CSV file for data
def check_required_columns(data):
    required_columns = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
    columns_present = all(column in data.columns for column in required_columns)
    
    # Find extra columns that are not in the required columns
    extra_columns = []
    for column in data.columns:
        if column not in required_columns:
            extra_columns.append(column)
    
    print(f"Do we have the relevant columns? {columns_present}")
    if extra_columns:
        print(f"Extra columns found: {extra_columns}")
    return columns_present

# Function to generate the cleaned file path
def generate_cleaned_file_path(file_path):
    base, ext = os.path.splitext(file_path)
    return f"{base}_cleaned{ext}"

# Function to clean the CSV data file
def clean_csv_data(file_path):
    cleaned_file_path = generate_cleaned_file_path(file_path)
    df = pd.read_csv(file_path, sep='\t', header=None, names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
    print(df.head())

    if check_required_columns(df):
        # Check if DateTime is already in the correct format
        try:
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            print("DateTime column is already in the correct format.")
            df.rename(columns={'DateTime': 'datetime'}, inplace=True)
        except ValueError:
            # Separate the combined datetime column into separate date and time columns
            df[['Date', 'Time']] = df['DateTime'].str.split(' ', expand=True)
            df.drop(columns=['DateTime'], inplace=True)
            # Combine Date and Time into a single datetime column
            df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            df.drop(columns=['Date', 'Time'], inplace=True)
            print("Date and Time columns combined into a single datetime column.")
        
        # Reorder the columns to have datetime in the first place
        df = df[['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
        # Save the cleaned data to a new CSV file
        df.to_csv(cleaned_file_path, index=False)
        print("Cleaned data saved to:", cleaned_file_path)
        cleaned_df = pd.read_csv(cleaned_file_path, parse_dates=['datetime'])
        print(cleaned_df.head())
        return cleaned_df
    else:
        print("The CSV file does not contain all required columns.")
        return None

# Call the function and clean the CSV data
cleaned_df = clean_csv_data(original_csv_path)