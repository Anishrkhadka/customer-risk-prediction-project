import pandas as pd
from datetime import datetime, timedelta

def read_sheets_from_excel_file(filename, sheet_name):
    """
    Reads a specific sheet from an Excel file (binary format).

    Args:
        filename (str): Path to the Excel file (.xlsb).
        sheet_name (str or int): Name or index of the sheet to read.

    Returns:
        pd.DataFrame: Data from the specified sheet.
    """
    return pd.read_excel(filename, sheet_name=sheet_name, engine="pyxlsb")

def describe_df(df, title="DataFrame"):
    """
    Provides a detailed summary of a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to describe.
        title (str, optional): Title to display for the summary. Defaults to "DataFrame".

    Displays:
        - Head of the DataFrame
        - Data types
        - Missing value counts
        - Unique value counts per column
    """
    print(f"\n--- {title} ---")
    
    print("\nHead:")
    display(df.head())

    print("\nData Types:")
    display(df.dtypes)

    print("\nMissing Values:")
    display(df.isnull().sum()[df.isnull().sum() > 0])

    print("\nUnique Value Counts:")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")

def convert_excel_serial_date_to_datetime(excel_serial):
    """
    Converts Excel serial date format to Python datetime.

    Args:
        excel_serial (float or int): Excel serial date.

    Returns:
        datetime or pd.NaT: Converted datetime object or NaT if input is null.
    """
    if pd.isnull(excel_serial):
        return pd.NaT
    return datetime(1899, 12, 30) + timedelta(days=int(excel_serial))

def save_df_as_csv(df, filesname, is_save_index=False):
    """
    Saves a DataFrame as a CSV file.

    Args:
        df (pd.DataFrame): Data to save.
        filesname (str): Destination file path.
        is_save_index (bool): Whether to include index in CSV. Defaults to False.
    """
    df.to_csv(filesname, index=is_save_index)

def fill_missing_categoricals(df, cat_cols):
    """
    Fills missing values in categorical columns and adds missing flags.

    Args:
        df (pd.DataFrame): Input DataFrame.
        cat_cols (list of str): List of categorical columns to fill.

    Returns:
        pd.DataFrame: Updated DataFrame with filled categories and missing flags.
    """
    for col in cat_cols:
        df[f'missing_{col}'] = df[col].isna()
        df[col] = df[col].fillna(f'MISSING_{col}')
    return df

def time_agg(df, value_col, agg_func='mean', date_col='invoice_date', freq='M', label=None, rolling=None):
    """
    Aggregates a value column over time with optional rolling average.

    Args:
        df (pd.DataFrame): Input data.
        value_col (str): Column to aggregate.
        agg_func (str): Aggregation function (e.g., 'mean', 'sum').
        date_col (str): Column with datetime values.
        freq (str): Pandas resample frequency string (e.g., 'M', 'W').
        label (str, optional): Label for the output value column.
        rolling (int, optional): Rolling window size.

    Returns:
        pd.DataFrame: Aggregated data with time-based periods.
    """
    df = df.copy()
    df['period'] = pd.to_datetime(df[date_col]).dt.to_period(freq).dt.to_timestamp()
    grouped = df.groupby('period')[value_col].agg(agg_func).reset_index()
    grouped.columns = ['period', label or value_col]

    if rolling:
        grouped[label or value_col] = grouped[label or value_col].rolling(rolling, min_periods=1).mean()

    return grouped
