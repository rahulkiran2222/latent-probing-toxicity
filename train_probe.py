import argparse
from pathlib import Path

import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def load_npz(path):
    """
    Expected format:

        labels
        layer_0
        layer_1
        layer_2
        ...

    Each layer must have shape:

        (number_of_examples, embedding_dimension)
    """

    data = np.load(path, allow_pickle=False)

    if "labels" not in data.files:
        raise ValueError("Missing 'labels' in NPZ file.")

    labels = np.asarray(data["labels"]).astype(int)

    layers = {
        name: np.asarray(data[name])
        for name in data.files
        if name.startswith("layer_")
    }

    if not layers:
        raise ValueError(
            "No layer embeddings found. "
            "Expected layer_0, layer_1, ..."
        )

    return layers, labels


def build_probe(C=1.0):
    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    max_iter=3000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_layer(X, y):
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_probe()

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

    return {
        "accuracy": accuracy_score(y_valid, predictions),
        "precision": precision_score(
            y_valid,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_valid,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_valid,
            predictions,
            zero_division=0,
        ),
    }


def cross_validate_layer(X, y):
    model = build_probe()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
    )

    return scores.mean(), scores.std()


def train_final(X, y, layer_name, output_path):
    model = build_probe()

    model.fit(X, y)

    payload = {
        "model": model,
        "layer": layer_name,
        "task": "toxicity_detection",
        "n_samples": len(y),
        "embedding_dimension": X.shape[1],
    }

    joblib.dump(
        payload,
        output_path,
    )

    print(f"\nSaved trained probe:")
    print(output_path)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
        help="Path to NPZ containing embeddings and labels.",
    )

    parser.add_argument(
        "--layer",
        default=None,
        help="Specific layer to train.",
    )

    parser.add_argument(
        "--output",
        default="trained_probe.joblib",
    )

    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Evaluate every available layer.",
    )

    args = parser.parse_args()

    layers, y = load_npz(args.data)

    print("\nDataset")
    print("-------")
    print(f"Samples: {len(y)}")
    print(f"Classes: {np.unique(y)}")
    print(f"Positive rate: {y.mean():.4f}")
    print(f"Layers: {len(layers)}")

    # ---------------------------------------------------------
    # Layer sweep
    # ---------------------------------------------------------

    if args.sweep:

        results = []

        print("\nLayer sweep")
        print("-----------")

        for layer_name in sorted(
            layers,
            key=lambda x: int(x.split("_")[1]),
        ):

            X = layers[layer_name]

            print(
                f"\n{layer_name}: "
                f"shape={X.shape}"
            )

            metrics = evaluate_layer(X, y)

            cv_mean, cv_std = cross_validate_layer(
                X,
                y,
            )

            result = {
                "layer": layer_name,
                **metrics,
                "cv_accuracy": cv_mean,
                "cv_std": cv_std,
            }

            results.append(result)

            print(
                f"accuracy={metrics['accuracy']:.4f} | "
                f"f1={metrics['f1']:.4f} | "
                f"CV={cv_mean:.4f} ± {cv_std:.4f}"
            )

        # Use CV accuracy as our main selection criterion.
        best = max(
            results,
            key=lambda r: r["cv_accuracy"],
        )

        print("\n==============================")
        print("BEST LAYER")
        print("==============================")

        for key, value in best.items():
            print(f"{key}: {value}")

        best_layer = best["layer"]

        train_final(
            layers[best_layer],
            y,
            best_layer,
            args.output,
        )

        return

    # ---------------------------------------------------------
    # Train selected layer
    # ---------------------------------------------------------

    if args.layer is None:
        raise ValueError(
            "Provide --layer or use --sweep."
        )

    if args.layer not in layers:
        raise ValueError(
            f"Unknown layer: {args.layer}\n"
            f"Available: {sorted(layers)}"
        )

    train_final(
        layers[args.layer],
        y,
        args.layer,
        args.output,
    )


if __name__ == "__main__":
    main()
