"""Leakage-safe data loading and classical text feature pipeline."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

LABELS = ["Common Cold", "allergy", "Pneumonia", "Bronchial Asthma", "Dengue", "Malaria", "Typhoid", "Chicken pox"]
LABELS = ["Common Cold", "Allergy", "Pneumonia", "Bronchial Asthma", "Dengue", "Malaria", "Typhoid", "Chicken pox"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "Symptom2Disease.csv"

@dataclass
class DatasetSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    X_train_pca: np.ndarray
    X_test_pca: np.ndarray
    train_text: np.ndarray
    test_text: np.ndarray
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    angle_scaler: MinMaxScaler
    pca_scaler: StandardScaler


def load_frame(path=DEFAULT_DATA):
    df = pd.read_csv(path)
    if not {"label", "text"}.issubset(df.columns):
        raise ValueError("CSV must contain 'label' and 'text' columns")
    df = df[["label", "text"]].dropna().copy()
    df["label"] = df.label.astype(str).str.strip()
    df["text"] = df.text.astype(str).str.strip()
    # Dataset uses lower-case 'allergy'; normalize only this known spelling.
    df["label"] = df.label.map(lambda s: "Allergy" if s.lower() == "allergy" else s)
    df = df[df.label.isin(LABELS)].drop_duplicates(subset=["label", "text"]).reset_index(drop=True)
    return df


def make_split(seed=0, path=DEFAULT_DATA, test_size=.25):
    df = load_frame(path)
    tr, te = train_test_split(np.arange(len(df)), test_size=test_size, random_state=seed,
                              stratify=df.label)
    train, test = df.iloc[tr], df.iloc[te]
    vec = TfidfVectorizer(stop_words="english", max_features=None)
    A = vec.fit_transform(train.text)
    B = vec.transform(test.text)
    k = min(8, A.shape[1] - 1, A.shape[0] - 1)
    svd = TruncatedSVD(n_components=k, n_iter=7, random_state=seed)
    U, V = svd.fit_transform(A), svd.transform(B)
    # Pad to 8 if tiny/custom input yields fewer components.
    U = np.pad(U, ((0,0),(0,8-k))); V = np.pad(V, ((0,0),(0,8-k)))
    angle_scaler = MinMaxScaler(feature_range=(0, np.pi)).fit(U)
    pca_scaler = StandardScaler().fit(U)
    return DatasetSplit(angle_scaler.transform(U), angle_scaler.transform(V),
        train.label.map(LABEL_TO_ID).to_numpy(), test.label.map(LABEL_TO_ID).to_numpy(),
        pca_scaler.transform(U), pca_scaler.transform(V), train.text.to_numpy(), test.text.to_numpy(),
        vec, svd, angle_scaler, pca_scaler)


def validation_indices(y, seed=0, fraction=.2):
    fit, val = train_test_split(np.arange(len(y)), test_size=fraction, random_state=seed, stratify=y)
    return fit, val


def nested_train_validation(split, seed=0, fraction=.2, view="angle"):
    """Build a strict fit/validation view with every text transform fit on fit rows only."""
    from sklearn.model_selection import train_test_split
    fit, val = train_test_split(np.arange(len(split.y_train)), test_size=fraction,
                                random_state=seed, stratify=split.y_train)
    vec = TfidfVectorizer(stop_words="english")
    a = vec.fit_transform(split.train_text[fit]); b = vec.transform(split.train_text[val])
    k = min(8, a.shape[1]-1, a.shape[0]-1)
    svd = TruncatedSVD(n_components=k, n_iter=7, random_state=seed)
    za, zb = svd.fit_transform(a), svd.transform(b)
    za=np.pad(za,((0,0),(0,8-k))); zb=np.pad(zb,((0,0),(0,8-k)))
    scaler = MinMaxScaler((0,np.pi)) if view=="angle" else StandardScaler()
    xa=scaler.fit_transform(za); xb=scaler.transform(zb)
    return xa, xb, split.y_train[fit], split.y_train[val], fit, val
