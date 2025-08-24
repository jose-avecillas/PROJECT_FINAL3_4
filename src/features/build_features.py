import pandas as pd

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna().copy()
