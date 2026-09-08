from pathlib import Path

import joblib
import numpy as np


class Classifier:
    """
    CodaBench inference interface.

    The trained model is created locally and stored as
    trained_probe.joblib.

    CodaBench calls:
        Classifier().predict(X)
    """

    def __init__(self):
        model_path = Path(__file__).resolve().parent / "trained_probe.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Could not find trained_probe.joblib at {model_path}"
            )

        payload = joblib.load(model_path)

        # Support either:
        # 1. a raw sklearn model
        # 2. a dictionary containing {"model": model}
        if isinstance(payload, dict) and "model" in payload:
            self.model = payload["model"]
        else:
            self.model = payload

    def predict(self, X):
        X = np.asarray(X)

        predictions = self.model.predict(X)

        return np.asarray(predictions, dtype=int)
