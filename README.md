# Latent Probing for Toxicity Detection

Research and competition repository for the TRI AI / CodaBench Latent Probe Challenge - Toxicity Detection.

## Goal

Train lightweight linear probes on Gemma hidden-state representations to determine which layers contain linearly decodable toxicity information.

## Current pipeline

```text
Gemma layer embeddings
        |
        v
preprocessing / normalization
        |
        v
linear probe (Logistic Regression)
        |
        v
toxicity prediction
        |
        v
F1 / precision / recall / accuracy
```

## Repository status

This repository is the **research/training scaffold**. The final CodaBench `classifier.py` wrapper will be adapted to the organizer's starter-kit API once the starter-kit format is available.

Do not upload the current repository ZIP directly to CodaBench yet.

## Expected data

The loader supports a simple `.npz` format:

- `layer_0`, `layer_1`, ...: arrays shaped `(n_samples, hidden_dim)`
- `labels`: binary labels shaped `(n_samples,)`

Example:

```python
np.savez(
    "data/embeddings.npz",
    layer_0=layer0,
    layer_1=layer1,
    labels=labels,
)
```

If the official starter kit uses another format, only the data-loading adapter should need to change.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --data data/embeddings.npz --layer layer_0
python -m src.evaluate --model models/probe_layer_0.joblib --data data/embeddings.npz --layer layer_0
```

## Research questions

1. Which Gemma layer gives the strongest toxicity signal?
2. How early is toxicity linearly decodable?
3. Does normalization improve probe performance?
4. Does a small linear probe remain effective under distribution shift?
5. How much performance can we obtain without fine-tuning Gemma?

## Important

The competition evaluator is the source of truth for the final submission format. The final submission should contain only the files required by CodaBench.
