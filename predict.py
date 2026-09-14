"""
CLI inference using the saved TF-IDF + Logistic Regression model (the
primary model — see README for why it was chosen over the BiLSTM).

Usage:
    python predict.py "I absolutely loved the new season, best show ever!"
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from preprocessing import clean_text

MODEL_DIR = Path(__file__).resolve().parent / "models"


def load_model() -> tuple[TfidfVectorizer, LogisticRegression]:
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    model = joblib.load(MODEL_DIR / "logreg_model.joblib")
    return vectorizer, model


def predict(
    text: str, vectorizer: TfidfVectorizer, model: LogisticRegression
) -> tuple[str, float]:
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    proba = model.predict_proba(vec)[0, 1]
    label = "Positive" if proba >= 0.5 else "Negative"
    return label, float(proba)


def explain_prediction(
    text: str,
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """Exact per-word/bigram contribution to the logit, for one prediction.

    Because the model is linear, `coefficient * tfidf_weight` for each
    present feature IS the model's reasoning — not an approximation like
    SHAP/LIME would need to produce for a non-linear model.
    """
    cleaned = clean_text(text)
    vec: spmatrix = vectorizer.transform([cleaned])
    feature_names = vectorizer.get_feature_names_out()

    contributions = [
        (feature_names[idx], float(vec[0, idx] * model.coef_[0, idx]))
        for idx in vec.nonzero()[1]
    ]
    contributions.sort(key=lambda pair: abs(pair[1]), reverse=True)
    return contributions[:top_n]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python predict.py "some tweet text"')
        sys.exit(1)

    input_text = " ".join(sys.argv[1:])
    tfidf, clf = load_model()
    sentiment_label, positive_proba = predict(input_text, tfidf, clf)
    top_contributions = explain_prediction(input_text, tfidf, clf)

    print(f"Text:       {input_text}")
    print(f"Sentiment:  {sentiment_label}")
    print(f"Confidence: {positive_proba:.3f} (probability of Positive)")
    print("Top contributing words/bigrams:")
    for token, weight in top_contributions:
        direction = "+ positive" if weight > 0 else "- negative"
        print(f"  {token!r:20s} {direction}  (contribution={weight:+.3f})")
