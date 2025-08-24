from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def load_raw(filename: str) -> pd.DataFrame:
    path = DATA_DIR / "raw" / filename
    return pd.read_csv(path)

def save_processed(df: pd.DataFrame, filename: str) -> None:
    path = DATA_DIR / "processed" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
