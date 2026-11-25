"""Export the TF-IDF + MultinomialNB pipeline to model.json for client-side inference."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from app import load_or_train_pipeline

BASE_DIR = Path(__file__).resolve().parent
MODEL_JSON_PATH = BASE_DIR / "model.json"


def export_model() -> None:
    pipeline = load_or_train_pipeline()
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    # Map vocabulary token -> { idf, log_prob_ham, log_prob_spam }
    tokens = {}
    for token, idx in sorted(tfidf.vocabulary_.items()):
        tokens[token] = {
            "idf": float(tfidf.idf_[idx]),
            "ham": float(clf.feature_log_prob_[0][idx]),
            "spam": float(clf.feature_log_prob_[1][idx]),
        }

    payload = {
        "version": "1.0.0",
        "model": "MultinomialNB",
        "classes": ["ham", "spam"],
        "class_log_prior": [float(x) for x in clf.class_log_prior_],
        "stop_words": sorted(ENGLISH_STOP_WORDS),
        "tokens": tokens,
    }

    MODEL_JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Exported {len(tokens)} tokens to {MODEL_JSON_PATH}")


if __name__ == "__main__":
    export_model()
