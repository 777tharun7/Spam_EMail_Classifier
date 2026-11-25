from __future__ import annotations

import os
import random
import re
import string
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Spam_Mail"
PIPELINE_PATH = MODEL_DIR / "spam_pipeline.joblib"
STATIC_DIR = BASE_DIR / "static"

SPAM_PHRASES = [
    "You've been chosen for an exclusive investment opportunity!",
    "Alert: Unusual login detected. Secure your account now.",
    "Earn cash daily with zero experience. Start today!",
    "Winner! Your email was selected for a prize.",
    "Hot deal! 90% off on premium gadgets for 24 hours.",
    "Re: Your invoice has a refund. Click to process.",
    "Final notice! Your subscription will expire soon.",
    "Claim your airline voucher before it expires.",
    "Act fast! Limited slots for free crypto training.",
    "Get your free smartwatch—just pay shipping!",
]

HAM_PHRASES = [
    "Can we reschedule the call to tomorrow afternoon?",
    "I uploaded the project files to the shared folder.",
    "Thanks for your help earlier, really appreciate it!",
    "Will you be joining the team dinner tonight?",
    "The presentation has been moved to 11 am.",
    "Just parked outside your building. Coming up now.",
    "Let’s review the code changes after lunch.",
    "Had a great time at the reunion—let’s catch up again!",
    "Meeting went well. I'll send you the summary notes.",
    "Heading out for a walk. Need a break from the screen.",
]


def clean_text(text: str) -> str:
    """Normalize text by lowercasing, stripping digits, and removing punctuation."""
    lowered = text.lower()
    no_digits = re.sub(r"\d+", "", lowered)
    no_punct = "".join(char for char in no_digits if char not in string.punctuation)
    return " ".join(no_punct.split())


def train_and_save_pipeline(path: Path = PIPELINE_PATH) -> Pipeline:
    """Train the TF-IDF + MultinomialNB pipeline and persist it to disk."""
    rng = random.Random(42)
    data = [("spam", rng.choice(SPAM_PHRASES)) for _ in range(5000)] + [
        ("ham", rng.choice(HAM_PHRASES)) for _ in range(5000)
    ]
    rng.shuffle(data)

    df = pd.DataFrame(data, columns=["label", "message"])
    df["cleaned_message"] = df["message"].apply(clean_text)
    y = df["label"].apply(lambda label: 0 if label == "ham" else 1)

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("clf", MultinomialNB()),
        ]
    )
    pipeline.fit(df["cleaned_message"], y)

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return pipeline


def load_or_train_pipeline() -> Pipeline:
    """Load the saved classification pipeline, or train one if missing."""
    if PIPELINE_PATH.exists():
        loaded = joblib.load(PIPELINE_PATH)
        if isinstance(loaded, Pipeline):
            return loaded
    return train_and_save_pipeline()


pipeline = load_or_train_pipeline()

app = FastAPI(
    title="Spam Email Classifier",
    description="TF-IDF + Multinomial Naive Bayes Spam/Ham classifier API",
    version="1.0.0",
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class PredictRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000, description="Email or message text")


class PredictResponse(BaseModel):
    label: str
    is_spam: bool
    spam_probability: float
    ham_probability: float
    cleaned_text: str
    matched_tokens: list[str]


@app.get("/")
def index() -> FileResponse:
    """Serve the web UI."""
    index_file = BASE_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(index_file)


@app.get("/styles.css")
def styles_css() -> FileResponse:
    return FileResponse(BASE_DIR / "styles.css", media_type="text/css")


@app.get("/script.js")
def script_js() -> FileResponse:
    return FileResponse(BASE_DIR / "script.js", media_type="application/javascript")


@app.get("/model.json")
def model_json() -> FileResponse:
    return FileResponse(BASE_DIR / "model.json", media_type="application/json")


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Return service health and model metadata."""
    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    return {
        "status": "ok",
        "model": "MultinomialNB",
        "vocabulary_size": len(vectorizer.vocabulary_),
    }


@app.get("/api/examples")
def examples() -> dict[str, list[dict[str, str]]]:
    """Return sample messages for quick testing in the UI."""
    return {
        "examples": [
            {"label": "Spam", "text": "Congratulations! You have won a free cruise trip. Claim your prize now!"},
            {"label": "Spam", "text": "Urgent! Unusual login detected. Verify your account immediately."},
            {"label": "Spam", "text": "Act fast! Limited slots for free crypto training and daily cash."},
            {"label": "Ham", "text": "Hey, don't forget our dinner plan at 8 pm tonight."},
            {"label": "Ham", "text": "I uploaded the project files to the shared folder. Let's review after lunch."},
            {"label": "Ham", "text": "Can we reschedule the call to tomorrow afternoon?"},
        ]
    }


@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    """Classify an email/message as Spam or Ham."""
    cleaned = clean_text(payload.message)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Message contains no classifiable text.")

    proba = pipeline.predict_proba([cleaned])[0]
    ham_prob = float(proba[0])
    spam_prob = float(proba[1])
    is_spam = spam_prob >= 0.5

    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    analyzer = vectorizer.build_analyzer()
    vocab = vectorizer.vocabulary_
    matched_tokens = sorted({tok for tok in analyzer(cleaned) if tok in vocab})

    return PredictResponse(
        label="Spam" if is_spam else "Ham",
        is_spam=is_spam,
        spam_probability=round(spam_prob, 4),
        ham_probability=round(ham_prob, 4),
        cleaned_text=cleaned,
        matched_tokens=matched_tokens,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=False)
