Absolutely. Here is the **complete copy-paste-ready `results/README.md`**. No extra explanation needed—just create `results/README.md` and paste this:

`markdown
# 📊 Results & Evaluation

This directory documents the experimental results obtained from the latent probing pipeline and the final CodaBench evaluation.

---

## 🏆 Evaluation Summary

| Evaluation | Accuracy |
|---|---:|
| 🧪 Local validation | **92.58%** |
| 🏁 CodaBench hidden evaluation | **~53%** |
| 🏆 CodaBench leaderboard position | **8th** |

---

## 🧪 Local Validation

The trained linear probe achieved:

> **92.58% validation accuracy**

during local evaluation.

The experiment used the Layer-14 latent representations extracted from Gemma-2-2B and trained a lightweight `LinearSVC` classifier on top of those frozen representations.

This demonstrates that the selected latent representation contains a strong linearly decodable signal for the classification setup used during local training.

---

## 🏁 Official CodaBench Evaluation

The final classifier was submitted to the **LATENT PROBE CHALLENGE — TOXICITY DETECTION** on CodaBench.

The official hidden evaluation produced an accuracy of approximately:

> **53%**

The recorded leaderboard position was:

> **#8**

The submitted inference package consisted of:


submission.zip
├── classifier.py
└── trained_probe.joblib
`

The evaluation environment performed prediction using the pre-trained classifier. No training was performed during the official inference stage.

---

## 📉 Local vs. Hidden Evaluation

The difference between local validation and the official hidden evaluation is substantial:


Local Validation
██████████████████▌  92.58%

CodaBench Hidden Evaluation
██████████▋          ~53%


This gap is an important experimental result rather than something to hide.

It indicates that the probe generalized much less effectively to the hidden evaluation distribution than to the data used during local validation.

---

## 🔍 Interpretation

The experiment highlights an important property of latent probing:

> **A representation may contain highly predictive information for one dataset while a probe trained on that dataset may generalize poorly to another distribution.**

The results therefore provide evidence for two observations:

### 1. Decodability

A simple linear classifier can extract useful classification signal from an intermediate Gemma-2-2B representation.

The local validation result of **92.58%** demonstrates strong linear separability under the local training setup.

### 2. Distribution Sensitivity

The substantially lower hidden-test performance demonstrates that probe performance is highly dependent on the relationship between the training and evaluation distributions.

A strong in-distribution result does not necessarily translate into strong out-of-distribution performance.

---

## ⚠️ Dataset & Distribution-Shift Limitation

The local training pipeline relied on an externally available dataset because the expected competition training data was not available through the participant workflow used during this experiment.

The externally sourced dataset was therefore used to construct the supervised training signal for the probe.

This introduces an important limitation:

> **The external training distribution may differ substantially from the competition's hidden evaluation distribution.**

As a result:

* The **92.58%** local validation accuracy should not be interpreted as an estimate of competition performance.
* The **~53%** CodaBench result is the relevant measurement of generalization to the hidden competition evaluation.
* The large performance gap provides evidence of distribution mismatch.

---

## 🧠 Why the Result Matters

The experiment is not simply about maximizing classification accuracy.

The broader objective is to investigate whether information related to the target property is encoded within an LLM's intermediate representations.

The workflow demonstrates:


                    ┌─────────────────┐
                    │    Input Text   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Gemma-2-2B    │
                    │     Frozen      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Layer 14     │
                    │ Hidden State    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Mean Pooling   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  LinearSVC      │
                    │      Probe      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Binary Output   │
                    │      0 / 1      │
                    └─────────────────┘


The base language model remains frozen throughout probe training.

---

## 🔬 Future Experiments

Several experiments could potentially improve generalization and provide deeper insight into the representation:

* [ ] Train using the official competition embedding distribution, if available
* [ ] Compare representations from Layers 5, 10, 14, 18 and 23
* [ ] Compare mean pooling with alternative pooling strategies
* [ ] Evaluate Logistic Regression as an alternative linear probe
* [ ] Explore different `C` regularization values
* [ ] Investigate class-balancing strategies
* [ ] Evaluate nonlinear probes
* [ ] Perform cross-dataset evaluation
* [ ] Visualize embeddings using PCA
* [ ] Visualize embeddings using UMAP
* [ ] Analyze how decodability changes across model layers
* [ ] Investigate robustness under paraphrasing and distribution shifts

---

## 📌 Reproducibility

The complete training pipeline is available in:


../train_probe.py


The CodaBench-compatible inference implementation is available in:


../classifier.py


The required Python dependencies are specified in:


../requirements.txt


The main repository documentation is available in:


../README.md


---

## 📈 Key Numbers


┌──────────────────────────────────────┐
│          EXPERIMENT SUMMARY          │
├──────────────────────────────────────┤
│ Model:        Gemma-2-2B             │
│ Probe:        LinearSVC               │
│ Layer:        14                      │
│ Pooling:      Mean                    │
│ Max tokens:   64                     │
│                                      │
│ Local:        92.58%                 │
│ Hidden:       ~53%                   │
│ Rank:         #8                     │
└──────────────────────────────────────┘


---

## 🎯 Final Takeaway

The project demonstrates a complete latent-probing workflow:


LLM
 ↓
Intermediate Representation
 ↓
Latent Embedding
 ↓
Linear Probe
 ↓
Prediction
 ↓
Evaluation


The strong local result shows that the target signal can be linearly decoded from the selected representation under the local training distribution.

The lower hidden-test result highlights the importance of **dataset alignment, distribution shift, and representation generalization** when evaluating probes on unseen data.

This makes the experiment useful not only as a classification task, but also as a practical investigation into the **information encoded within intermediate language-model representations**.

---

<div align="center">

### 🧠 Decode the Representation

### 🔍 Measure the Signal

### 📊 Understand the Generalization Gap

</div>


This is the version I recommend keeping as the **final `results/README.md`**.
