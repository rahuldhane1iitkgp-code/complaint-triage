"""Single entry point for the app: raw complaint text in, predictions out.

Serves the TF-IDF + Logistic Regression pipelines from notebook 03. That is a
deliberate choice, not a shortcut: on the held-out evaluation DistilBERT was
statistically indistinguishable from these models (paired bootstrap interval
spanning zero), while costing ~75x the compute and 500MB of weights. See the
README for the numbers.
"""
import os
from functools import lru_cache

import joblib
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .preprocessing import clean_text

TASKS = ("product", "issue")
MODEL_DIR = os.environ.get(
    "MODEL_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
)

# VADER's own published thresholds. NOT tuned on this data - the tone signal is
# reported as a descriptive indicator and was never validated against a label,
# because CFPB carries no sentiment ground truth.
VADER_POS, VADER_NEG = 0.05, -0.05


@lru_cache(maxsize=1)
def load_models():
    models = {}
    for task in TASKS:
        path = os.path.join(MODEL_DIR, f"tfidf_lr_{task}.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Download tfidf_lr_{task}.joblib from notebook 03's "
                f"output and place it in {MODEL_DIR}/."
            )
        models[task] = joblib.load(path)
    return models


@lru_cache(maxsize=1)
def load_vader():
    return SentimentIntensityAnalyzer()


def _classify(pipe, cleaned, top_k):
    proba = pipe.predict_proba([cleaned])[0]
    classes = pipe.classes_
    order = np.argsort(proba)[::-1][:top_k]
    return {
        "label": str(classes[order[0]]),
        "confidence": float(proba[order[0]]),
        "top_k": [(str(classes[i]), float(proba[i])) for i in order],
    }


def tone(raw_text):
    """VADER on the RAW text - capitalisation and punctuation carry intensity,
    so this must not receive the cleaned string."""
    scores = load_vader().polarity_scores(str(raw_text))
    c = scores["compound"]
    label = "Positive" if c >= VADER_POS else "Negative" if c <= VADER_NEG else "Neutral"
    return {"label": label, "compound": float(c), "scores": scores, "validated": False}


def predict(text, top_k=3):
    """Raw complaint text -> product, issue, confidences, and an unvalidated tone."""
    raw = str(text or "")
    cleaned = clean_text(raw)
    n_words = len(cleaned.split())

    if n_words < 5:
        return {
            "ok": False,
            "reason": "Needs at least 5 words after cleaning to classify. "
                      "Redaction markers and punctuation are stripped, so very "
                      "short or heavily redacted text can fall below the floor.",
            "n_words": n_words,
        }

    models = load_models()
    out = {"ok": True, "n_words": n_words, "cleaned": cleaned,
           "tone": tone(raw)}
    for task in TASKS:
        out[task] = _classify(models[task], cleaned, top_k)
    return out


def predict_batch(texts, top_k=1):
    """Vectorised path for CSV upload - one transform per task, not one per row."""
    raw = [str(t or "") for t in texts]
    cleaned = [clean_text(t) for t in raw]
    ok = np.array([len(c.split()) >= 5 for c in cleaned])

    models = load_models()
    rows = {"n_words": [len(c.split()) for c in cleaned], "classified": ok}

    for task in TASKS:
        labels = np.full(len(cleaned), "", dtype=object)
        conf = np.full(len(cleaned), np.nan)
        if ok.any():
            sub = [c for c, k in zip(cleaned, ok) if k]
            proba = models[task].predict_proba(sub)
            classes = models[task].classes_
            labels[ok] = classes[proba.argmax(1)]
            conf[ok] = proba.max(1)
        rows[f"{task}"] = labels
        rows[f"{task}_confidence"] = conf

    t = [tone(r) for r in raw]
    rows["tone"] = [x["label"] for x in t]
    rows["tone_compound"] = [x["compound"] for x in t]
    return rows
