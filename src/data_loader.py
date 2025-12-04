import os
import polars as pl  # Polars for faster CSV handling
import pandas as pd  # Optional: for conversion to Pandas

from typing import Optional, List

# Dynamically determine the project's root directory
# Project root is the parent directory of the 'src' folder
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Columns required by the preprocessing pipeline
# This dramatically reduces memory usage (from ~20GB to ~2GB)
REQUIRED_COLUMNS = [
    # Core identifiers and target
    "FPA_ID",
    "FIRE_SIZE",
    "FIRE_YEAR",
    "DISCOVERY_DOY",
    "NWCG_CAUSE_CLASSIFICATION",
    "LATITUDE",
    "LONGITUDE",
    "STATE",  # Added for entity embeddings
    # Risk management
    "SDI",
    # Fire stations
    "No_FireStation_20.0km",
    # GACC preparedness
    "GACC_PL",
    # Global human modification
    "GHM",
    # NDVI
    "NDVI-1day",
    # National preparedness level
    "NPL",
    # Social vulnerability
    "EPL_PCI",
    # Rangeland
    "rpms",
    "rpms_1km",
]

# GRIDMET columns (5-day window features) - these contain '_5D_' in their names
GRIDMET_PATTERNS = ["_5D_"]


def _get_gridmet_columns(all_columns: List[str]) -> List[str]:
    """Extract GRIDMET columns (containing '_5D_') from column list."""
    return [col for col in all_columns if any(pattern in col for pattern in GRIDMET_PATTERNS)]


def load_data_from_csv(convert_to_pandas: bool = True, use_subset: bool = True) -> Optional[pd.DataFrame]:
    """
    Load the FPA FOD dataset from CSV.

    Args:
        convert_to_pandas: If True, convert to Pandas DataFrame
        use_subset: If True, only load required columns (memory-efficient)

    Returns:
        DataFrame with fire data, or None if loading fails
    """
    file_path = os.path.join(project_root, 'data/external/FPA_FOD_Plus.csv')
    file_path = os.path.abspath(file_path)

    if not os.path.exists(file_path):
        print(f"File not found. Please make sure the file exists at: {file_path}")
        return None

    try:
        if use_subset:
            # Memory-efficient: First scan to get column names, then read only needed columns
            print("Scanning CSV for column names...")

            # Read just the header to get column names
            header_df = pl.read_csv(file_path, n_rows=0)
            all_columns = header_df.columns

            # Get GRIDMET columns
            gridmet_cols = _get_gridmet_columns(all_columns)

            # Combine required columns + GRIDMET columns
            columns_to_load = list(set(REQUIRED_COLUMNS + gridmet_cols))
            # Filter to only columns that exist
            columns_to_load = [c for c in columns_to_load if c in all_columns]

            print(f"Loading {len(columns_to_load)} columns (out of {len(all_columns)} total)...")

            # Read only the needed columns
            polars_df = pl.read_csv(
                file_path,
                columns=columns_to_load,
                infer_schema_length=10000,
                ignore_errors=True
            )
        else:
            # Original behavior: load all columns
            print("Loading the file using Polars (all columns)...")
            polars_df = pl.read_csv(file_path, infer_schema_length=10000, ignore_errors=True)

        print(f"File loaded successfully with Polars. Shape: {polars_df.shape}")

        if convert_to_pandas:
            print("Converting to Pandas...")
            # Convert to pandas - this should now work with reduced memory
            pandas_df = polars_df.to_pandas()
            # Explicitly delete polars df to free memory
            del polars_df
            print(f"Conversion complete. Pandas shape: {pandas_df.shape}")
            return pandas_df
        else:
            return polars_df

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    df = load_data_from_csv()
    if df is not None:  # Check if the file was loaded successfully
        print(df.head())
    else:
        print("Data frame could not be loaded. Exiting the program.")
