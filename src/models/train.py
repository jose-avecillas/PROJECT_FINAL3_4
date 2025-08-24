import argparse, joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import pandas as pd

from src.data.load import load_raw
from src.features.build_features import basic_clean

MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "trained_models"

parser = argparse.ArgumentParser()
parser.add_argument("--raw_file", default="train.csv")
parser.add_argument("--target", default="target")

def train_and_eval(df: pd.DataFrame, target: str = "target"):
    df = basic_clean(df)
    X = df.drop(columns=[target])
    y = df[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid={"n_estimators": [100, 300], "max_depth": [None, 10, 20]},
        cv=5, n_jobs=-1, scoring="f1_weighted",
    )
    grid.fit(Xtr, ytr)
    ypred = grid.predict(Xte)
    metrics = {"accuracy": float(accuracy_score(yte, ypred)),
               "f1_weighted": float(f1_score(yte, ypred, average="weighted"))}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(grid.best_estimator_, MODELS_DIR / "model.pkl")
    return metrics, MODELS_DIR / "model.pkl"

if __name__ == "__main__":
    args = parser.parse_args()
    df = load_raw(args.raw_file)
    metrics, model_path = train_and_eval(df, target=args.target)
    print({"metrics": metrics, "model_path": str(model_path)})
