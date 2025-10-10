import os
import polars as pl  # Polars for faster CSV handling
import pandas as pd  # Optional: for conversion to Pandas

from typing import Optional

# Dynamically determine the project's root directory
# Project root is the parent directory of the 'src' folder
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_data_from_csv(convert_to_pandas: bool = True) -> Optional[pd.DataFrame]:
    # Construct the correct path to the CSV file
    file_path = os.path.join(project_root, 'data/external/FPA_FOD_Plus.csv')
    # Ensure we deal with absolute paths
    file_path = os.path.abspath(file_path)

    # Check if the file exists
    if os.path.exists(file_path):
        try:
            # Use Polars to read the CSV efficiently
            print("Loading the file using Polars...")
            polars_df = pl.read_csv(file_path, infer_schema_length=10000, ignore_errors=True)
            print(f"File loaded successfully with Polars. Shape: {polars_df.shape}")

            if convert_to_pandas:
                pandas_df = polars_df.to_pandas()
                print("Converted the Polars DataFrame to Pandas.")
                return pandas_df
            else:
                return polars_df
        except Exception as e:
            # Handle other potential errors
            print(f"An error occurred while reading the file: {e}")
            return None
    else:
        print(f"File not found. Please make sure the file exists at: {file_path}")
        return None


if __name__ == '__main__':
    df = load_data_from_csv()
    if df is not None:  # Check if the file was loaded successfully
        print(df.head())
    else:
        print("Data frame could not be loaded. Exiting the program.")
