import argparse, json, joblib
import pandas as pd
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "trained_models"

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="Ruta a JSON con features")
parser.add_argument("--model", default=str(MODELS_DIR / "model.pkl"))

def run_inference(payload: dict, model_path: str = str(MODELS_DIR / "model.pkl")):
    model = joblib.load(model_path)
    X = pd.DataFrame([payload])
    y = model.predict(X)
    return y.tolist()

if __name__ == "__main__":
    args = parser.parse_args()
    payload = json.load(open(args.input))
    print({"prediction": run_inference(payload, args.model)})
