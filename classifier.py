"""
Local inference wrapper for a trained latent toxicity probe.

This is deliberately independent of the CodaBench API because the official
starter kit/evaluator interface has not yet been provided.

Before CodaBench submission, adapt the `predict()` function to the exact
interface required by the official starter kit.
"""

import joblib
import numpy as np


class ToxicityClassifier:
    def __init__(self, probe_path):
        payload = joblib.load(probe_path)
        self.model = payload["model"]
        self.layer = payload.get("layer")

    def predict(self, embeddings):
        """
        embeddings: numpy array of shape (N, hidden_dim)
        returns: numpy array of binary predictions
        """
        X = np.asarray(embeddings)
        return self.model.predict(X)

    def predict_proba(self, embeddings):
        X = np.asarray(embeddings)
        return self.model.predict_proba(X)[:, 1]


def load_classifier(probe_path):
    return ToxicityClassifier(probe_path)
