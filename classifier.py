from pathlib import Path
import joblib
import numpy as np


class Classifier:
    def __init__(self):
        model_path = Path(__file__).resolve().parent / "trained_probe.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing {model_path}")
        self.model = joblib.load(model_path)

    def predict(self, X):
        X = np.asarray(X)
        pred = self.model.predict(X)
        return np.asarray(pred, dtype=int)
