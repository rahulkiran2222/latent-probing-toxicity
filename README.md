# Latent Probing for Toxicity Detection

Flat-file starter implementation for the TRI AI / CodaBench competition.

## Files

- `train_probe.py` — trains a linear toxicity probe from layer embeddings.
- `classifier.py` — local inference wrapper for a saved probe.
- `requirements.txt` — Python dependencies.

## Important

The CodaBench starter kit was not available in the participant page shown to us. Therefore `classifier.py` is intentionally written as a local inference wrapper and must be adapted to the official CodaBench evaluator API before final submission.

Do not submit this repository ZIP to CodaBench until the official starter-kit interface is confirmed.

## Expected embedding file

The training script expects an NPZ containing:

- `labels`: binary labels, shape `(N,)`
- `layer_0`, `layer_1`, ...: hidden-state embeddings, shape `(N, D)`

Example:

```python
import numpy as np
np.savez(
    "embeddings.npz",
    labels=labels,
    layer_0=layer0,
    layer_1=layer1,
)
```

## Train

```bash
pip install -r requirements.txt
python train_probe.py --data embeddings.npz --layer layer_0 --output probe.joblib
```

## Sweep layers

```bash
python train_probe.py --data embeddings.npz --sweep
```
