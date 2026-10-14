import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import sklearn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, create_model
from sklearn.datasets import load_breast_cancer

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
MODEL_CARD_PATH = Path(
    os.getenv("MODEL_CARD_PATH", "artifacts/model_card.json")
)
THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))

FEATURE_NAMES = [
    name.replace(" ", "_")
    for name in load_breast_cancer().feature_names
]


class PredictionBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


PredictionInput = create_model(
    "PredictionInput",
    __base__=PredictionBase,
    **{
        name: (float, Field(..., ge=0, allow_inf_nan=False))
        for name in FEATURE_NAMES
    },
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.is_file():
        logger.error("Model artifact not found: %s", MODEL_PATH)
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run the DVC training pipeline."
        )

    if not MODEL_CARD_PATH.is_file():
        logger.error("Model card not found: %s", MODEL_CARD_PATH)
        raise RuntimeError(f"Model card not found at {MODEL_CARD_PATH}")

    app.state.model = joblib.load(MODEL_PATH)

    with MODEL_CARD_PATH.open("r", encoding="utf-8") as file:
        app.state.model_card = json.load(file)

    logger.info("Model loaded from %s", MODEL_PATH)
    yield


app = FastAPI(
    title="Breast Cancer Triage Service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictionInput):
    values = payload.model_dump()
    feature_order = app.state.model_card["feature_names"]

    row = [[values[name.replace(" ", "_")] for name in feature_order]]

    probabilities = app.state.model.predict_proba(row)[0]
    classes = list(app.state.model.classes_)
    malignant_index = classes.index(1)
    malignant_probability = float(probabilities[malignant_index])

    label = (
        "malignant"
        if malignant_probability >= THRESHOLD
        else "benign"
    )

    return {
        "label": label,
        "malignant_probability": round(malignant_probability, 4),
        "threshold": THRESHOLD,
        "model_version": app.state.model_card.get("model_version", "1.0.0"),
    }


@app.get("/model/info")
def model_info():
    card = app.state.model_card

    return {
        "model_version": os.getenv(
            "MODEL_VERSION", card.get("model_version", "1.0.0")
        ),
        "model_family": card.get("model_family"),
        "scikit_learn_version": card.get(
            "scikit_learn_version", sklearn.__version__
        ),
        "git_commit": os.getenv("GIT_SHA", card.get("git_commit", "unknown")),
        "threshold": THRESHOLD,
        "feature_count": card.get("feature_count", len(FEATURE_NAMES)),
        "test_metrics": card.get("test_metrics", {}),
    }


@app.get("/stats")
def stats():
    # Redis request counting will be added in Milestone 4.
    return {"count": 0}