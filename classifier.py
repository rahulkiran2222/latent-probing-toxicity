import numpy as np


class Classifier:
    """
    Basic starting-kit example.

    This does NOT do any real learning -- it always predicts class 0,
    regardless of input. It exists purely to show the required interface
    (a class named Classifier with a predict method) so you know what
    your own submission needs to look like.

    Replace this with your own trained probe:
      1. Extract Gemma embeddings for your training data.
      2. Train a classifier (e.g. logistic regression) on those embeddings.
      3. Save your trained model (e.g. with joblib.dump).
      4. Load it here in __init__, and use it in predict() instead of the
         constant-zero placeholder below.
    """

    def __init__(self):
        pass

    def predict(self, X):
        X = np.asarray(X)
        return np.zeros(X.shape[0], dtype=int)
