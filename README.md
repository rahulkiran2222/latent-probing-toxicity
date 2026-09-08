# Latent Probe Challenge — Fast Training Pipeline

This version trains a binary linear probe for the competition task using
Gemma-2-2B layer-14 mean-pooled embeddings.

Competition specification used:
- model: google/gemma-2-2b
- layer: 14
- maximum sequence length: 64
- mean pooling across non-padding tokens
- output: 0/1

Because the competition page does not expose the public training data in the
participant UI, this pipeline uses a public mental-health classification corpus
as a training source and maps Normal -> 0 and all distress-related classes -> 1.

## Colab

```bash
!pip install -r requirements.txt
!python train_probe.py --max_samples 12000 --batch_size 8
```

The 12k run is a fast first experiment. Once it works, run the full corpus:

```bash
!python train_probe.py --batch_size 8
```

If the first run was interrupted after embeddings were created:

```bash
!python train_probe.py --reuse_embeddings --batch_size 8
```

After training, the submission ZIP must contain at its root:

- classifier.py
- trained_probe.joblib

Do not put these files inside another folder.
