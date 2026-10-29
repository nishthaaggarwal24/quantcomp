"""Interactive symptom sentence demo backed by saved preprocessing and QNN weights."""
import argparse, json
from pathlib import Path
import joblib
import numpy as np
from .data import LABELS
from .qnn import QNNConfig, VariationalClassifier

ROOT=Path(__file__).resolve().parents[1]

def predict(sentence, artifact=ROOT/"results"/"demo_model.joblib"):
    obj=joblib.load(artifact)
    x=obj["vectorizer"].transform([sentence]); z=obj["svd"].transform(x)
    model=obj["scaler"].transform(z)
    if model.shape[1] < 8: model=np.pad(model,((0,0),(0,8-model.shape[1])))
    qnn=VariationalClassifier(QNNConfig(**obj["config"])); qnn.weights=np.asarray(obj["weights"])
    probs=qnn.predict_proba(model)[0]
    return LABELS[int(probs.argmax())], {LABELS[i]:float(probs[i]) for i in range(8)}

def main():
    p=argparse.ArgumentParser(description="Experimental symptom text classifier (not a medical tool)")
    p.add_argument("sentence",nargs="?",help="symptoms in plain English")
    p.add_argument("--json",action="store_true",help="print a machine-readable result for the web demo")
    a=p.parse_args(); sentence=a.sentence or input("Describe symptoms: ")
    label,probs=predict(sentence)
    if a.json:
        print(json.dumps({"predicted":label,"probabilities":probs}))
        return
    print(f"Predicted dataset label: {label}")
    for disease,prob in sorted(probs.items(),key=lambda kv:-kv[1]): print(f"{disease:20s} {prob:.3f}")

if __name__=="__main__": main()
