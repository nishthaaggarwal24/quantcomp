"""Classical reference models on the same eight-dimensional features."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier


def make_model(name, seed=0):
    if name == "logistic": return LogisticRegression(max_iter=2000, random_state=seed)
    if name == "linear_svm": return LinearSVC(C=1.0, random_state=seed)
    if name == "rbf_svm": return SVC(kernel="rbf", C=10.0, gamma="scale", random_state=seed)
    if name == "mlp": return MLPClassifier(hidden_layer_sizes=(3,), max_iter=2000, learning_rate_init=.01, random_state=seed, early_stopping=False)
    if name == "tfidf_logistic": return LogisticRegression(max_iter=2000, random_state=seed)
    raise ValueError(name)


def parameter_count(model):
    if isinstance(model, SVC): return None  # nonlinear kernel has no finite weight count
    if hasattr(model, "coef_"):
        return int(model.coef_.size + model.intercept_.size)
    return int(sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_))


def evaluate_classical(seed=0, path=None):
    from .data import make_split, load_frame
    d = make_split(seed, path) if path else make_split(seed)
    results=[]
    for name in ("logistic", "linear_svm", "mlp", "rbf_svm"):
        model=make_model(name, seed).fit(d.X_train, d.y_train)
        results.append({"model":name,"accuracy":float(model.score(d.X_test,d.y_test)),"parameters":parameter_count(model)})
    # Uncompressed TF-IDF baseline, still fit on training data only.
    frame=load_frame(path) if path else load_frame()
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd
    tr,te=train_test_split(np.arange(len(frame)), test_size=.25, random_state=seed, stratify=frame.label)
    vec=TfidfVectorizer(stop_words="english"); a=vec.fit_transform(frame.text.iloc[tr]); b=vec.transform(frame.text.iloc[te])
    from .data import LABEL_TO_ID
    ytr=frame.label.iloc[tr].map(LABEL_TO_ID).to_numpy(); yte=frame.label.iloc[te].map(LABEL_TO_ID).to_numpy()
    model=make_model("tfidf_logistic",seed).fit(a,ytr)
    results.append({"model":"tfidf_logistic","accuracy":float(model.score(b,yte)),"parameters":int(model.coef_.size+model.intercept_.size)})
    return results
