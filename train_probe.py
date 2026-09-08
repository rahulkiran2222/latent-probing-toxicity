from pathlib import Path
import argparse
import gc
import json

import joblib
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from transformers import AutoModel, AutoTokenizer


DATASET_ID = "ourafla/Mental-Health_Text-Classification_Dataset"
DATA_FILE = "mental_heath_unbanlanced.csv"
MODEL_ID = "google/gemma-2-2b"
LAYER = 14
MAX_LENGTH = 64


def load_training_data(max_samples=None, seed=42):
    path = hf_hub_download(
        repo_id=DATASET_ID,
        filename=DATA_FILE,
        repo_type="dataset",
    )
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "status"]).copy()

    # Competition task description says:
    # 0 = normal/safe, 1 = mental-health distress signal.
    df["label"] = (df["status"].str.lower() != "normal").astype(np.int64)

    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text"])

    if max_samples is not None and max_samples < len(df):
        # Keep the binary classes stratified when taking a faster subset.
        _, df = train_test_split(
            df,
            test_size=max_samples,
            stratify=df["label"],
            random_state=seed,
        )

    return df.reset_index(drop=True)


def extract_layer14_embeddings(texts, batch_size=8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading {MODEL_ID} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        output_hidden_states=True,
    )
    model.to(device)
    model.eval()

    chunks = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True)
            hidden = outputs.hidden_states[LAYER]

            # Mean over real (non-padding) tokens.
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        chunks.append(pooled.float().cpu().numpy())

        if (start // batch_size) % 50 == 0:
            print(f"Embedded {min(start + batch_size, len(texts))}/{len(texts)}")

    embeddings = np.concatenate(chunks, axis=0)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings


def find_best_probe(X_train, y_train, X_val, y_val):
    candidates = []

    # Raw embeddings
    for C in [0.03, 0.1, 0.3, 1.0, 3.0]:
        for class_weight in [None, "balanced"]:
            model = LinearSVC(
                C=C,
                class_weight=class_weight,
                max_iter=10000,
                dual="auto",
                random_state=42,
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            acc = accuracy_score(y_val, pred)
            candidates.append((acc, f"raw | C={C} | weight={class_weight}", model))
            print(f"{candidates[-1][1]:45s} accuracy={acc:.5f}")

    # Standardized embeddings
    for C in [0.03, 0.1, 0.3, 1.0, 3.0]:
        for class_weight in [None, "balanced"]:
            model = Pipeline([
                ("scale", StandardScaler()),
                ("svc", LinearSVC(
                    C=C,
                    class_weight=class_weight,
                    max_iter=10000,
                    dual="auto",
                    random_state=42,
                )),
            ])
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            acc = accuracy_score(y_val, pred)
            candidates.append((acc, f"scaled | C={C} | weight={class_weight}", model))
            print(f"{candidates[-1][1]:45s} accuracy={acc:.5f}")

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Use a stratified subset for a faster first run.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--reuse_embeddings", action="store_true")
    args = parser.parse_args()

    df = load_training_data(max_samples=args.max_samples)
    print("\nDataset:")
    print(df["status"].value_counts())
    print(df["label"].value_counts())

    emb_path = Path("layer14_embeddings.npz")

    if args.reuse_embeddings and emb_path.exists():
        data = np.load(emb_path)
        X = data["X"]
        y = data["y"]
        print(f"Loaded cached embeddings: {X.shape}")
    else:
        X = extract_layer14_embeddings(
            df["text"].tolist(),
            batch_size=args.batch_size,
        )
        y = df["label"].to_numpy(dtype=np.int64)
        np.savez_compressed(emb_path, X=X, y=y)
        print(f"Saved embeddings to {emb_path}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    print(f"\nTrain: {X_train.shape}")
    print(f"Validation: {X_val.shape}")

    best_acc, best_name, best_model = find_best_probe(
        X_train, y_train, X_val, y_val
    )

    print(f"\nBEST VALIDATION MODEL: {best_name}")
    print(f"Validation accuracy: {best_acc:.5f}")

    # Retrain the selected model on all available training data.
    # Reconstruct the same estimator from its name.
    parts = best_name.split("|")
    mode = parts[0].strip()
    C = float(parts[1].split("=")[1].strip())
    weight_text = parts[2].split("=")[1].strip()
    class_weight = None if weight_text == "None" else "balanced"

    if mode == "raw":
        final_model = LinearSVC(
            C=C,
            class_weight=class_weight,
            max_iter=10000,
            dual="auto",
            random_state=42,
        )
    else:
        final_model = Pipeline([
            ("scale", StandardScaler()),
            ("svc", LinearSVC(
                C=C,
                class_weight=class_weight,
                max_iter=10000,
                dual="auto",
                random_state=42,
            )),
        ])

    final_model.fit(X, y)
    joblib.dump(final_model, "trained_probe.joblib", compress=3)

    report = {
        "dataset": DATASET_ID,
        "model": MODEL_ID,
        "layer": LAYER,
        "max_length": MAX_LENGTH,
        "n_samples": int(len(y)),
        "embedding_dim": int(X.shape[1]),
        "best_validation_accuracy": float(best_acc),
        "best_probe": best_name,
    }
    Path("probe_report.json").write_text(json.dumps(report, indent=2))

    print("\nDONE.")
    print("Created: trained_probe.joblib")
    print("Created: probe_report.json")
    print("\nUse trained_probe.joblib + classifier.py for the CodaBench ZIP.")


if __name__ == "__main__":
    main()
