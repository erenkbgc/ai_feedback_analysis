
import pandas as pd

def clean_text(df: pd.DataFrame, text_column: str = 'reviews.text') -> pd.DataFrame:
    """
    Cleans review text: removes NaN, trims whitespace, drops empty.

    Args:
        df (pd.DataFrame): Input DataFrame.
        text_column (str): Name of the text column.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df = df.copy()
    df = df.dropna(subset=[text_column])
    df[text_column] = df[text_column].str.strip()
    df = df[df[text_column] != ""]
    return df

def clean_ratings(df, rating_column='reviews.rating'):
    import re

    def extract_number(x):
        if pd.isnull(x):
            return None
        match = re.search(r'(\d+(\.\d+)?)', str(x))
        if match:
            return float(match.group(1))
        else:
            return None

    df[rating_column] = df[rating_column].apply(extract_number)
    df = df[df[rating_column].notnull()]
    df[rating_column] = df[rating_column].astype(int)
    return df
