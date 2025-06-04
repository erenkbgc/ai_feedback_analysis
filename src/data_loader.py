import pandas as pd
from typing import List, Optional

def load_data(
    filepath: str,
    usecols: Optional[List[str]] = None,
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Loads and optionally samples a CSV into a pandas DataFrame.

    Args:
        filepath (str): Path to the CSV file.
        usecols (List[str], optional): Specific columns to load.
        sample_size (int, optional): If provided, randomly sample this many rows.
        random_state (int): Random state for reproducibility.

    Returns:
        pd.DataFrame: Loaded (and optionally sampled) DataFrame.
    """
    df = pd.read_csv(filepath, usecols=usecols)
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(sample_size, random_state=random_state).reset_index(drop=True)
    return df

def save_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Saves a DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        filepath (str): Destination path.
    """
    df.to_csv(filepath, index=False)
