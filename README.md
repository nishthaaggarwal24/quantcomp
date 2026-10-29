# Symptom text classification with a variational quantum circuit

This project studies an eight-class symptom-text classifier. A TF-IDF/TruncatedSVD pipeline produces eight classical features, which are encoded into a Qiskit Aer circuit. A trainable variational circuit returns probabilities for the eight possible measurement outcomes. Classical models are evaluated on the same seeded splits as references.

**This is an educational experiment, not a medical tool. It must not be used for diagnosis or treatment.** The dataset contains short synthetic symptom descriptions and the held-out test set is small.

## Classes and data

The included `data/Symptom2Disease.csv` is the supplied 1,200-row source file. We keep Common Cold, Allergy, Pneumonia, Bronchial Asthma, Dengue, Malaria, Typhoid, and Chicken pox. Deduplication leaves 388 unique examples (291 train and 97 test at 75/25). The split is stratified and deterministic for a seed; seeds 0–4 provide repeated splits. TF-IDF, TruncatedSVD, and feature scalers are fit on training data only. TF-IDF vocabulary size and the variance explained by eight SVD components are data/split dependent and are recorded by the run, rather than hard-coded.

The text vectorization, SVD, scaling, final argmax, and optimizer are classical. The quantum part is feature encoding, the trainable circuit, and measurement. Eight SVD dimensions retain limited information; the compression is a central limitation. Amplitude encoding normalizes the vector and therefore discards its norm.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -e '.[dev]'
```

## Run

```bash
python -m src.runner                 # classical baselines, results/classical.json
python -m src.runner tune            # 19 phase-A structures; saves after each run
python -m src.runner final           # selected configs, test evaluation, plots
python -m src.runner demo-model      # (see below) train a model for the sentence demo
python -m src.predict "I have a high fever, chills and night sweats"
python -m pytest -q
```

Run durations for QNN training can be substantial: parameter shift evaluates two shifted circuits per trainable parameter and sample batch. Aer is a simulator; this is not a hardware runtime estimate. The amplitude circuit is transpiled by Aer to its supported basis before simulation to avoid controlled-gate simulator inconsistencies.

## Pipeline and circuit choices

1. TF-IDF removes English stop words and creates sparse word-score features.
2. TruncatedSVD compresses to eight values.
3. Angle scaling maps values to [0, π]; amplitude scaling standardizes values.
4. Angle encoding applies one RY per feature (8 qubits). ZZ encoding adds pairwise phase interactions (8 qubits). Amplitude encoding prepares a normalized 8-amplitude vector (3 qubits).
5. The ansatz uses trainable RY/RZ rotations and a CNOT ring. There are `2*n_qubits*(depth+1)` weights. Re-uploading adds another encoding before each ansatz layer for angle/ZZ.
6. Exact simulator probabilities are read for qubits 0–2, giving eight classes. No finite-shot noise is used by default.
7. Training uses cross-entropy, parameter-shift gradients, Adam mini-batches, and validation early stopping.

## Experiment flow

Phase A compares angle/ZZ at depths 1 and 3, re-upload on/off, and bandwidth 1.0/0.5, plus amplitude depths 1–3. The script persists `results/tuning_A.json` incrementally. Select/tune only with validation metrics; `runner final` trains from scratch and evaluates each selected encoding on the held-out set once per seed. Do not inspect test scores to choose a configuration. `results/classical.json`, `results/final_evaluation.json`, and generated PNGs contain run outputs. Results may differ from illustrative estimates because package versions, seeds, and splits matter.

## Report honestly

Classical logistic regression and SVMs are expected to outperform this small QNN. Compare accuracy and parameter counts, but do not claim quantum advantage or speedup. Report the 8-feature compression, classical text preprocessing, simulator-only execution, small test set, and synthetic text distribution. A larger raw-TF-IDF logistic baseline illustrates the information lost during compression.

## References

- Havlíček et al., “Supervised learning with quantum-enhanced feature spaces,” Nature 567, 209–212 (2019), DOI: 10.1038/s41586-019-0980-2.
- Pérez-Salinas et al., “Data re-uploading for a universal quantum classifier,” Quantum 4, 226 (2020), DOI: 10.22331/q-2020-02-06-226.
- Shaydulin & Wild, “Importance of kernel bandwidth in quantum machine learning,” Phys. Rev. A 106, 042407 (2022), DOI: 10.1103/PhysRevA.106.042407.

## Results from this workspace run

All results below use seeds 0–4 and are reported as mean ± sample standard deviation of test accuracy. The QNN configuration for each encoding was selected using Phase A validation accuracy, then evaluated once on each seed's test split. The same three selected circuit configurations were used across all five splits. Phase A and final QNN training used five epochs; the final models were trained on each full training split. The test split was not used for configuration selection.

| Model | Parameters | Test accuracy |
|---|---:|---:|
| Logistic regression, 8 features | 72 | 87.0% ± 2.6 |
| Linear SVM, 8 features | 72 | 84.1% ± 3.3 |
| Tiny neural network (8→3→8) | 59 | 64.3% ± 18.6 |
| RBF SVM, 8 features | — | 85.8% ± 2.0 |
| Logistic regression, full TF-IDF | 4,432 | 94.6% ± 2.1 |
| QNN, angle (depth 3, re-upload, bandwidth 0.5) | 64 | 10.5% ± 1.1 |
| QNN, ZZ (depth 1, re-upload, bandwidth 0.5) | 32 | 15.3% ± 3.0 |
| QNN, amplitude (depth 2) | 18 | 10.1% ± 3.1 |

The QNN results are near or below the 12.5% random-guess rate. Classical models substantially outperform them in this experiment. This run provides no evidence of quantum advantage. Counts for full TF-IDF depend on the fitted vocabulary and software version. Detailed records and plots are under `results/`.


## Present the running model

The project includes a standalone Streamlit app with live prediction, a pipeline explanation, and the measured results table. From macOS Terminal (with the project environment activated), run:

```bash
python -m pip install -e '.[demo]'
streamlit run app.py
```

Streamlit opens the app in your browser, usually at `http://localhost:8501`. Keep the Terminal window open while presenting; press Ctrl+C there to stop the app. The saved model is `results/demo_model.joblib`. The app runs locally and does not require Codex.
