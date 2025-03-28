import pandas as pd
import os
from apy import *

# Define the correct column order
EXPECTED_COLUMNS = ["datetime", "open", "high", "low", "close"]

def load_csv(file_path):
    print("1) Loading the CSV file and try to detect the delimiter automatically.")
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"❌ Error loading the file: {e}")
        return None

def check_header(df):
    print("2) Checking if the first row contains the expected column names.")
    actual_columns = list(df.columns)
    print(actual_columns)

    if actual_columns == EXPECTED_COLUMNS:
        print("✅ The columns are already in the correct order.")
        return True

    # Check missing columns and extra ones
    ''' senior way : missing_columns = [col for col in EXPECTED_COLUMNS if col not in actual_columns]
    senior way : extra_columns = [col for col in actual_columns if col not in EXPECTED_COLUMNS] '''
    missing_columns = []
    for col in EXPECTED_COLUMNS:
        if col not in actual_columns:
            missing_columns.append(col)

    extra_columns = []
    for col in actual_columns:
        if col not in EXPECTED_COLUMNS:
            extra_columns.append(col)
    
    if missing_columns:
        print(f"⚠️ Warning: Missing columns: {missing_columns}")

    if extra_columns:
        print(f"ℹ️ Notice: Extra columns detected (will be ignored): {extra_columns}")

    # Ask for user confirmation if required columns are missing
    if missing_columns:
        user_input = input("Some required columns are missing. Do you want to continue? (yes/no): ").strip().lower()
        if user_input != "yes":
            print("❌ Operation canceled.")
            return False
        print("✅ Continuing despite missing columns...")
    return True

def rename_first_column_to_datetime(df):
    """Ensure the first column is named 'datetime'."""
    if df.columns[0].lower() != "datetime":
        print(f"🔄 Renaming the first column '{df.columns[0]}' to 'datetime'.")
        df.columns = ["datetime"] + list(df.columns[1:])
    else:
        print("✅ The first column is already named 'datetime'.")
    return df


def reorder_columns(df):
    """Display current column order and allow user to reorder them."""
    print("3) Current column order detected:")
    
    # Display original columns with their indexes
    for i, col in enumerate(df.columns):
        print(f"{col} ({i})", end=", ")
    print("\n")  # New line for better readability

    # Ask user for the new order
    user_input = input("4) Enter the new column order as comma-separated indexes (0,1,3,2,4): ").strip()
    
    try:
        new_order = [int(i) for i in user_input.split(",")]
        if sorted(new_order) != list(range(len(df.columns))):
            print("❌ Invalid column order. Please enter all indexes in a valid sequence.")
            return df  # Return original DataFrame if the input is incorrect
        
        # Apply new order
        new_columns = [df.columns[i] for i in new_order]
        df = df[new_columns]
        print("✅ Columns have been rearranged successfully.")

    except ValueError:
        print("❌ Invalid input. Please enter numeric indexes separated by commas.")
    
    return df

def fix_header_format(df):
    print("5) Formating the header: lowercase.")
    df.columns = [col.lower() for col in df.columns]
    print("✅ Header formatting fixed.")
    return df

def add_missing_header(df):
    """If the first row looks like data (only numbers), add the correct header."""
    if not all(isinstance(col, str) for col in df.columns):  # If columns are numbers
        df.columns = EXPECTED_COLUMNS  # Set the correct header
        print("✅ Header was missing and has been added.")

    return df

def save_csv(df, output_file):
    print("6) Saving the cleaned DataFrame to a new CSV file.")
    df.to_csv(output_file, index=False)
    print(f"✅ Cleaned file saved as: {output_file}")

cleaned_csv_file = None  # Global variable to store the cleaned CSV file path

def process_csv():
    global cleaned_csv_file
    file_path = gold_daily  # Change the CSV file path here

    if not os.path.exists(file_path):
        print("❌ File not found. Please check the path.")
        return None

    df = load_csv(file_path)
    if df is None:
        return None

    if not check_header(df):
        return None

    df = rename_first_column_to_datetime(df)
    df = reorder_columns(df)
    df = fix_header_format(df)
    df = add_missing_header(df)

    cleaned_csv_file = "cleaned_" + os.path.basename(file_path)
    save_csv(df, cleaned_csv_file)

    return cleaned_csv_file

# Automatically process the CSV file when the script is imported
process_csv()
