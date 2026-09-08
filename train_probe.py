import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_data(path):
    data = np.load(path, allow_pickle=False)
    if "labels" not in data.files:
        raise ValueError("NPZ must contain a 'labels' array.")

    layers = {k: data[k] for k in data.files if k.startswith("layer_")}
    if not layers:
        raise ValueError("NPZ must contain layer_0, layer_1, ... arrays.")

    return layers, data["labels"].astype(int)


def make_probe():
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])


def run_layer(name, X, y, output=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = make_probe()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    }

    print(
        f"{name}: "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}"
    )

    if output:
        joblib.dump({
            "model": model,
            "layer": name,
            "task": "toxicity_detection",
            "metrics": metrics,
        }, output)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--layer")
    parser.add_argument("--output", default="probe.joblib")
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    layers, y = load_data(args.data)

    if args.sweep:
        results = {}
        for name in sorted(layers):
            results[name] = run_layer(name, layers[name], y)

        best = max(results, key=lambda k: results[k]["f1"])
        print(f"BEST LAYER: {best} | F1={results[best]['f1']:.4f}")
        return

    if args.layer not in layers:
        raise ValueError(
            f"Unknown layer {args.layer!r}. Available: {sorted(layers)}"
        )

    run_layer(args.layer, layers[args.layer], y, args.output)


if __name__ == "__main__":
    main()
