"""Presentation-ready browser demo for the symptom-text QNN project."""
from pathlib import Path
import json
import subprocess
import sys

import pandas as pd
import streamlit as st

from src.data import LABELS

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def run_prediction(sentence):
    """Run Aer in a child process so native simulator libraries stay isolated from Streamlit."""
    completed = subprocess.run(
        [sys.executable, "-m", "src.predict", "--json", sentence],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=180,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "Prediction process failed")
    return json.loads(completed.stdout)


def read_json(name):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def show_plot(title, filename, description, width="stretch"):
    path = RESULTS / filename
    st.markdown(f"#### {title}")
    if path.exists():
        st.image(str(path), width=width)
        st.caption(description)
    else:
        st.info(f"{filename} has not been generated yet.")


st.set_page_config(page_title="Symptom Classifier | QNN Demo", page_icon="🧪", layout="wide")
st.markdown(
    """
    <style>
      :root {
        --page: #f8f5f0;
        --surface: #fffdfa;
        --surface-muted: #f4eee8;
        --ink: #242421;
        --muted: #827d76;
        --line: #edc6b7;
        --line-strong: #d7a08b;
        --accent: #e46643;
        --accent-dark: #ca5032;
        --accent-soft: #fff0e9;
        --quantum: #e46643;
        --quantum-soft: #fff0e9;
        --radius: 10px;
      }
      .stApp {background:var(--page); color:var(--ink); font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
      .block-container {max-width:1360px; padding:3.75rem 2rem 4rem;}
      [data-testid="stHeader"] {background:var(--page); border-bottom:1px solid var(--line);}
      h1,h2,h3,h4 {color:var(--ink); line-height:1.25;}
      h1,h2,h3,h4,.hero h1 {font-family:Georgia,"Times New Roman",serif; letter-spacing:-.02em;}
      .block-container h1,.block-container h2,.block-container h3,.block-container h4 {font-family:Georgia,"Times New Roman",serif !important;}
      h1 {font-size:1.8rem; font-weight:700;}
      h2 {font-size:1.5rem; font-weight:700;}
      h3 {font-size:1.2rem; font-weight:700;}
      h4 {font-size:1rem; font-weight:700;}
      p,li {line-height:1.55; color:var(--ink);}
      .hero {display:grid; grid-template-columns:minmax(320px,1fr) minmax(270px,.9fr); align-items:center; gap:2rem;
             min-height:94px; padding:.85rem 0; border:0; border-bottom:1px solid var(--line); border-radius:0; margin:.2rem 0 1rem; background:transparent; color:var(--ink);}
      .hero h1 {align-self:center; max-width:none; font-size:1.85rem; line-height:1.12; margin:0; padding:0; color:var(--accent-dark);}
      .hero p {display:block; align-self:center; max-width:440px; margin:0 0 0 auto; padding:.2rem 0 .2rem 1.2rem;
               border:0; border-left:1px solid var(--line-strong); border-radius:0; background:transparent; color:var(--muted); font-size:.94rem; line-height:1.5;}
      [data-testid="stAlertContainer"] {background:var(--surface) !important; border-radius:7px; border:1px solid var(--line); box-shadow:none;}
      [data-testid^="stAlertContent"] {background:var(--surface) !important;}
      [data-testid="stAlertContainer"] p {color:var(--ink);}
      [data-testid="stTabs"] [role="tablist"] {gap:.45rem; padding:.55rem 0; border-top:1px solid var(--line); border-bottom:1px solid var(--line);}
      [data-testid="stTab"] {padding:.52rem .9rem; border:1px solid transparent; border-radius:999px; color:var(--muted); font-size:.88rem; font-weight:520; transition:background-color .15s ease,border-color .15s ease,color .15s ease;}
      [data-testid="stTab"]:hover {border-color:var(--line); color:var(--ink);}
      [data-testid="stTab"][aria-selected="true"] {color:var(--accent-dark); border-color:var(--accent); background:var(--surface);}
      [data-testid="stTab"] .react-aria-SelectionIndicator {display:none;}
      .block-container h3 {padding-bottom:.45rem; margin-top:1.8rem; border-bottom:1px solid var(--line);}
      [data-testid="stMetric"] {background:var(--surface); padding:.8rem .9rem .7rem; border:1px solid var(--line); border-radius:var(--radius);}
      [data-testid="stMetricLabel"] p {color:var(--muted); font-size:.84rem;}
      [data-testid="stMetricValue"] {color:var(--ink); font-weight:650;}
      [data-testid="stMetricDelta"] {color:var(--accent-dark) !important;}
      [data-testid="stMetricDelta"] svg {color:var(--accent-dark) !important;}
      [data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input {
        background:var(--surface); border:1px solid var(--line-strong); border-radius:7px; color:var(--ink);
      }
      [data-testid="stTextArea"] textarea:focus, [data-testid="stTextInput"] input:focus {
        border-color:var(--accent); box-shadow:0 0 0 2px rgba(228,102,67,.12);
      }
      button[kind="primaryFormSubmit"], button[data-testid^="stBaseButton-primary"] {background:var(--accent); border-color:var(--accent); color:#fff;}
      button[kind="primaryFormSubmit"] *, button[data-testid^="stBaseButton-primary"] * {color:#fff !important;}
      button[kind="primaryFormSubmit"]:hover, button[data-testid^="stBaseButton-primary"]:hover {background:var(--accent-dark); border-color:var(--accent-dark); color:#fff;}
      [data-testid="stFormSubmitButton"] button, [data-testid="stButton"] button {border-radius:7px; font-weight:560; transition:background-color .15s ease,border-color .15s ease;}
      [data-testid="stFormSubmitButton"] button:hover, [data-testid="stButton"] button:hover {border-color:var(--accent);}
      [data-testid="stDataFrame"] {border:1px solid var(--line); border-radius:7px; overflow:hidden; background:var(--surface);}
      [data-testid="stExpander"] {border:1px solid var(--line); border-radius:var(--radius); background:var(--surface);}
      [data-testid="stImage"] img {max-width:100%; height:auto; border:1px solid var(--line); border-radius:7px;}
      .step {height:100%; border:1px solid var(--line); border-radius:var(--radius); padding:1rem; background:var(--surface); min-height:135px;}
      .step small {color:var(--muted); text-transform:uppercase; letter-spacing:.06em; font-weight:650;}
      .step strong {display:block; margin:.4rem 0; font-size:1rem; color:var(--ink);}
      .step p {margin:0; color:var(--muted); font-size:.92rem;}
      .pipeline-group {border:1px solid var(--line); border-radius:var(--radius); padding:.85rem 1rem .3rem; margin:.65rem 0 .8rem; background:var(--surface);}
      .pipeline-group.classical {border-top:1px solid var(--line-strong);}
      .pipeline-group.quantum {border-top:2px solid var(--accent);}
      .pipeline-group-title {display:flex; align-items:center; gap:.55rem; margin:0 0 .5rem; color:var(--ink); font-size:1rem; font-weight:680;}
      .pipeline-step {display:grid; grid-template-columns:36px minmax(155px,205px) 1fr; gap:.75rem; align-items:center; padding:.65rem .35rem; border-top:1px solid var(--line);}
      .pipeline-number {display:grid; place-items:center; width:26px; height:26px; border:1px solid var(--line); border-radius:6px; background:var(--surface-muted); color:var(--ink); font-size:.82rem; font-weight:650;}
      .quantum .pipeline-number {background:var(--accent-soft); color:var(--accent);}
      .pipeline-title {font-weight:650; color:var(--ink);}
      .pipeline-detail {color:var(--muted); font-size:.92rem; line-height:1.5;}
      .pipeline-handoff {text-align:center; color:var(--accent); font-size:.88rem; font-weight:650; padding:.15rem 0 .25rem;}
      .tag {display:inline-block; border-radius:999px; padding:.22rem .55rem; margin:.1rem; background:var(--surface); border:1px solid var(--line); color:var(--muted); font-size:.85rem;}
      @media (max-width: 760px) {
        .block-container {padding:4rem 1rem 2.5rem;}
        .hero {grid-template-columns:minmax(0,1fr) minmax(210px,.85fr); min-height:82px; gap:1rem;}
        .hero h1 {font-size:1.55rem;}
        .hero p {padding-left:.85rem;}
        .pipeline-step {grid-template-columns:32px minmax(0,1fr); gap:.5rem .65rem; align-items:start;}
        .pipeline-detail {grid-column:2;}
        [data-testid="stTabs"] [role="tablist"] {overflow-x:auto; flex-wrap:nowrap;}
        [data-testid="stTab"] {padding:.65rem .75rem; white-space:nowrap;}
      }
      @media (max-width: 600px) {
        .hero {grid-template-columns:1fr; min-height:0; gap:.6rem; padding:.75rem 0;}
        .hero h1 {max-width:100%; font-size:1.6rem; padding:.15rem 0;}
        .hero p {max-width:none; margin:0; padding:.45rem 0 0; border-left:0; border-top:1px solid var(--line);}
      }
      @media (max-width: 420px) {
        .block-container {padding-left:.75rem; padding-right:.75rem;}
        .hero {grid-template-columns:1fr; min-height:0; gap:.6rem; padding:.75rem 0;}
        .hero h1 {max-width:100%; font-size:1.5rem; padding:.15rem 0;}
        .hero p {max-width:none; margin:0; padding:.45rem 0 0; border-left:0; border-top:1px solid var(--line);}
        .pipeline-group {padding:.7rem .7rem .25rem;}
        .pipeline-title {font-size:.94rem;}
        .pipeline-detail {font-size:.88rem;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero"><h1>Symptoms → quantum classifier</h1>'
    '<p>An interactive, honest comparison of a variational quantum classifier with classical models.</p></div>',
    unsafe_allow_html=True,
)
st.warning(
    "For teaching and research demonstration only. This model is not a medical device and must not be used for diagnosis or treatment decisions.",
    icon="⚠️",
)

demo_tab, method_tab, results_tab = st.tabs(["Try the model", "How it works", "Results & analysis"])

with demo_tab:
    st.subheader("Try a symptom description")
    st.write("Enter one sentence. The saved text pipeline transforms it, then the selected QNN produces scores for the eight dataset labels.")
    left, right = st.columns([0.82, 1.18], gap="large")
    with left:
        with st.form("symptom_form"):
            sentence = st.text_area(
                "Symptom description",
                placeholder="For example: I have a high fever, chills and I keep sweating at night.",
                height=145,
                max_chars=1200,
            )
            submitted = st.form_submit_button("Run the model", type="primary", width="stretch")
        st.caption("This demo uses the saved angle-encoding model. Its label scores are not medical probabilities.")
        if submitted:
            if not sentence.strip():
                st.error("Enter a symptom description first.")
            else:
                with st.spinner("Preparing text features and simulating the quantum circuit…"):
                    try:
                        result = run_prediction(sentence.strip())
                        st.session_state["prediction"] = result["predicted"]
                        st.session_state["probabilities"] = result["probabilities"]
                    except FileNotFoundError:
                        st.error("The demo model is missing. Run `python -m src.runner demo-model` first.")
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")
    with right:
        st.markdown("#### Model output")
        if "prediction" in st.session_state:
            disease = st.session_state["prediction"]
            probs = st.session_state["probabilities"]
            st.metric("Highest scoring dataset label", disease, f"{probs[disease]:.1%} model score")
            chart = pd.DataFrame({"Label": list(probs), "Score": list(probs.values())})
            chart = chart.sort_values("Score", ascending=True).set_index("Label")
            st.bar_chart(chart, horizontal=True, height=360, color="#e46643")
            st.caption("The eight scores sum to 100% because they are circuit outcome probabilities. They do not represent clinical certainty.")
        else:
            st.info("Your result and the full eight-label score chart will appear here.")

with method_tab:
    st.subheader("How the classifier works")
    st.write("The project turns a sentence into a quantum state, applies trainable gates, and maps three measured qubits to eight disease labels.")
    metrics = st.columns(4)
    for col, value, label in zip(metrics, ["388", "8", "8", "5"], ["unique sentences", "dataset labels", "SVD features", "stratified splits"]):
        col.metric(label, value)
    st.markdown("### The full pipeline")
    st.write("Read top to bottom. The text is prepared classically first; only then does it enter the quantum circuit.")
    classical_steps = [
        ("Sentence → TF-IDF", "Convert the sentence into word scores; common English stop words are removed."),
        ("569 → 8 features", "TruncatedSVD compresses the vocabulary vector. These eight dimensions retain about 18.1% of variance."),
        ("Scale the features", "Fit the scaler on training data, then prepare values as rotation angles or normalized amplitudes."),
    ]
    quantum_steps = [
        ("Encode the input", "Angle and ZZ use eight qubits; amplitude encoding represents eight values with three qubits."),
        ("Apply trainable circuit", "RY/RZ rotations and a CNOT ring transform the encoded quantum state."),
        ("Measure → classify", "Three measured qubits give eight outcome probabilities. A classical argmax maps the highest score to a disease label."),
    ]

    def pipeline_group(title, css_class, offset, items):
        rows = "".join(
            f'<div class="pipeline-step"><div class="pipeline-number">{offset + i}</div>'
            f'<div class="pipeline-title">{step_title}</div><div class="pipeline-detail">{detail}</div></div>'
            for i, (step_title, detail) in enumerate(items)
        )
        st.markdown(
            f'<div class="pipeline-group {css_class}"><div class="pipeline-group-title">{title}</div>{rows}</div>',
            unsafe_allow_html=True,
        )

    pipeline_group("Classical text processing", "classical", 1, classical_steps)
    st.markdown('<div class="pipeline-handoff">8 scaled features enter the circuit ↓</div>', unsafe_allow_html=True)
    pipeline_group("Quantum circuit and readout", "quantum", 4, quantum_steps)
    st.markdown("### Three encoding choices")
    enc_cols = st.columns(3, gap="medium")
    encoding_cards = [
        ("angle", "Angle · 8 qubits", "One feature controls one RY rotation. Simple and separable at encoding time."),
        ("zz", "ZZ feature map · 8 qubits", "Single-feature phases plus pairwise interactions entangle features. Bandwidth changes the angle scale."),
        ("amplitude", "Amplitude · 3 qubits", "A 3-qubit state stores eight normalized amplitudes; normalization discards the vector's original length."),
    ]
    for col, (encoding, title, detail) in zip(enc_cols, encoding_cards):
        with col:
            st.markdown(f"#### {title}")
            st.write(detail)
            selected = read_json("qnn_summary.json") or {}
            cfg = selected.get(encoding, {}).get("config", {})
            if cfg:
                reuse = "on" if cfg.get("reupload") else "off"
                st.caption(f"Selected: depth {cfg.get('depth')}, bandwidth {cfg.get('bandwidth')}, re-uploading {reuse}.")
    st.markdown("### Training and evaluation")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**Trainable circuit**")
        st.write("Each ansatz layer applies trainable RY/RZ rotations on every qubit followed by a CNOT ring. A final rotation layer follows the repeated layers. Data re-uploading can encode the input again between layers for angle and ZZ circuits.")
        st.markdown("**Optimization**")
        st.write("Cross-entropy loss is optimized with Adam mini-batches of 32. For each rotation weight, parameter shift estimates its gradient as [f(w + π/2) − f(w − π/2)] / 2. Validation accuracy selects the best checkpoint during tuning.")
    with b:
        st.markdown("**Fairness and leakage control**")
        st.write("The corpus is deduplicated before stratified seeded splits. TF-IDF, SVD, and scaling are fitted on training data and applied to held-out data. Five seeds provide repeated test estimates.")
        st.markdown("**What is quantum here?**")
        st.write("Text preparation and the final choice of highest score are classical. The encoding gates, trainable variational circuit, and measurement probabilities run in Qiskit Aer simulation. This is not a quantum speedup claim.")
    st.info("The ansatz has 2 × qubits × (depth + 1) trainable rotation weights. Readout uses exact simulator probabilities for the 8 computational-basis outcomes, without finite-shot sampling.")
    with st.expander("Project details and caveats"):
        st.write("The dataset has 24 original disease labels; this experiment keeps eight selected labels and 388 unique sentences. The 8-feature compression retains a small share of text variance. Results are sensitive to the small dataset and split choice. Simulator results do not establish performance on quantum hardware.")
        st.write("An Aer controlled-gate issue affected amplitude encoding during development. The circuit is transpiled to basic gates before simulation, and a regression test guards that workaround.")
        st.write("Labels: " + " · ".join(LABELS))

with results_tab:
    classical = read_json("classical_summary.json")
    qnn = read_json("qnn_summary.json")
    if not classical or not qnn:
        st.info("Run the experiments first to populate this page.")
    else:
        model_rows = [
            ("Logistic regression · 8 features", classical["logistic"]["accuracy_mean"], classical["logistic"]["accuracy_std"], "Classical"),
            ("Linear SVM · 8 features", classical["linear_svm"]["accuracy_mean"], classical["linear_svm"]["accuracy_std"], "Classical"),
            ("Tiny neural network · 8 features", classical["mlp"]["accuracy_mean"], classical["mlp"]["accuracy_std"], "Classical"),
            ("RBF SVM · 8 features", classical["rbf_svm"]["accuracy_mean"], classical["rbf_svm"]["accuracy_std"], "Classical"),
            ("Logistic regression · full TF-IDF", classical["tfidf_logistic"]["accuracy_mean"], classical["tfidf_logistic"]["accuracy_std"], "Classical · no compression"),
        ]
        for name, result in qnn.items():
            model_rows.append((f"Variational QNN · {name.title()}", result["mean_accuracy"], result["std_accuracy"], "Quantum circuit · simulator"))
        top_classical = classical["logistic"]["accuracy_mean"]
        best_qnn_name, best_qnn = max(qnn.items(), key=lambda item: item[1]["mean_accuracy"])
        kpis = st.columns(4)
        best_qnn_label = "ZZ" if best_qnn_name == "zz" else best_qnn_name.title()
        kpis[0].metric("Best QNN", f"{best_qnn['mean_accuracy']:.1%}", best_qnn_label)
        kpis[1].metric("Logistic · 8 features", f"{top_classical:.1%}")
        kpis[2].metric("Full TF-IDF logistic", f"{classical['tfidf_logistic']['accuracy_mean']:.1%}")
        kpis[3].metric("Random baseline", "12.5%", "8 balanced labels")
        st.markdown("### Test accuracy across five splits")
        st.write("The classical models lead in this experiment. The full TF-IDF baseline also shows what is lost by compressing text to eight features.")
        table_rows = [{"Model": n, "Mean accuracy": f"{m:.1%}", "Across-split variation": f"± {s:.1%}", "Type": kind} for n, m, s, kind in model_rows]
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch")
        st.caption("Mean ± sample standard deviation across five seeded held-out test splits. The standard deviation describes split-to-split variation, not a confidence interval.")

        st.markdown("### Performance by encoding")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            show_plot("Selected QNN configurations", "accuracy_by_encoding.png", "Five held-out scores per encoding. Configurations were selected using validation data; points represent different random splits.")
        with c2:
            show_plot("Accuracy by ansatz depth", "accuracy_by_depth.png", "Depth groups summarize the selected QNNs. Since encoding and other settings also differ, this is not a controlled depth-only comparison.")

        st.markdown("### What happened during training?")
        show_plot("Training cross-entropy", "training_curves.png", "Training loss from the first final-evaluation split for each selected encoding. Falling loss indicates training-set fit; held-out accuracy is the separate generalization measure.")

        st.markdown("### Which labels were confused?")
        st.write("Each matrix uses row-normalized percentages: each true-label row sums to 100%. The diagonal is correct classification; off-diagonal cells show where examples went instead.")
        matrix_tabs = st.tabs(["Angle", "ZZ", "Amplitude"])
        for tab, enc in zip(matrix_tabs, ["angle", "zz", "amplitude"]):
            with tab:
                show_plot(f"{enc.title()} encoding · normalized confusion matrix", f"confusion_{enc}.png", "Percent of held-out examples for each true label, aggregated across the five test splits.", width=850)

        with st.expander("How to read this comparison"):
            st.write("The QNNs are simulator-based research models, not optimized replacements for classical classifiers. Their scores are close to the random baseline in this run. The full TF-IDF model has many more features and parameters, so it is a useful information-retention reference rather than an equal-size model comparison.")
