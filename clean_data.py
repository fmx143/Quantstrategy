import pandas as pd
import os
from apy import *

# Define the correct column order
EXPECTED_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]

def load_csv(file_path):
    """Load the CSV file and try to detect the delimiter automatically."""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"❌ Error loading the file: {e}")
        return None

def check_header(df):
    """Check if the first row contains the expected column names."""
    actual_columns = list(df.columns)
    print(actual_columns)

    if actual_columns == EXPECTED_COLUMNS:
        print("✅ The columns are already in the correct order.")
        return True

    # Check if all required columns exist
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in actual_columns]
    if missing_columns:
        print(f"⚠️ Warning: Missing columns: {missing_columns}")
        user_input = input("Do you want to continue? (yes/no): ").strip().lower()
        if user_input != "yes":
            print("❌ Operation canceled.")
            return False

    return True

def reorder_columns(df):
    """Rearrange the columns to match the expected order."""
    df = df.copy()  # Avoid modifying the original DataFrame

    # Reorder columns by keeping the ones that exist, and adding missing ones as empty
    reordered_df = df.reindex(columns=EXPECTED_COLUMNS)

    print("✅ Columns have been reordered.")
    return reordered_df

def fix_header_format(df):
    """Format the header: lowercase except for the first letter."""
    df.columns = [col.capitalize() for col in df.columns]
    print("✅ Header formatting fixed.")
    return df

def add_missing_header(df):
    """If the first row looks like data (only numbers), add the correct header."""
    if not all(isinstance(col, str) for col in df.columns):  # If columns are numbers
        df.columns = EXPECTED_COLUMNS  # Set the correct header
        print("✅ Header was missing and has been added.")

    return df

def save_csv(df, output_file):
    """Save the cleaned DataFrame to a new CSV file."""
    df.to_csv(output_file, index=False)
    print(f"✅ Cleaned file saved as: {output_file}")

def main():
    """Main function to process the CSV file."""
    file_path = input(uj_15min).strip() # change the CSV file path here

    if not os.path.exists(file_path):
        print("❌ File not found. Please check the path.")
        return

    df = load_csv(file_path)
    if df is None:
        return

    if not check_header(df):
        return

    df = reorder_columns(df)
    df = fix_header_format(df)
    df = add_missing_header(df)

    output_file = "cleaned_" + os.path.basename(file_path)
    save_csv(df, output_file)
