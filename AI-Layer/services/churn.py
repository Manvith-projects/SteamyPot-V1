"""
Churn Prediction Service Router
================================
Wraps the Churn-Prediction project as a FastAPI APIRouter.

Endpoint:  POST /api/churn/predict
"""
import os
import json
import numpy as np
import pandas as pd
import joblib

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services import get_service_dir

router = APIRouter(prefix="/api/churn", tags=["Churn Prediction"])

SERVICE_DIR = get_service_dir("Churn-Prediction")
OUTPUT_DIR = os.path.join(SERVICE_DIR, "outputs")

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_model = None
_transformer = None
_feature_names = None
_initialized = False
_error = None

# ---------------------------------------------------------------------------
# Feature definitions (from original app.py)
# ---------------------------------------------------------------------------
NUMERIC_RAW = [
    "orders_last_30d", "orders_last_90d", "avg_order_value",
    "days_since_last_order", "order_frequency", "cancellation_rate",
    "avg_delivery_delay_min", "avg_user_rating", "num_complaints",
    "discount_usage_rate", "app_sessions_per_week",
    "preferred_order_hour", "account_age_days",
]

ENGINEERED = [
    "engagement_decay", "order_value_frequency",
    "experience_score", "recency_frequency", "complaint_rate",
]

ALL_FEATURES = NUMERIC_RAW + ENGINEERED


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------
def init():
    global _model, _transformer, _feature_names, _initialized, _error
    try:
        _model = joblib.load(os.path.join(OUTPUT_DIR, "best_model.joblib"))
        _transformer = joblib.load(os.path.join(OUTPUT_DIR, "transformer.joblib"))
        with open(os.path.join(OUTPUT_DIR, "feature_names.json")) as f:
            _feature_names = json.load(f)
        _initialized = True
        print(f"  [churn] Loaded model ({len(_feature_names)} features)")
    except Exception as e:
        _error = str(e)
        print(f"  [churn] FAILED: {e}")


# ---------------------------------------------------------------------------
# Helpers (from original app.py)
# ---------------------------------------------------------------------------
def _engineer(raw: dict) -> dict:
    feat = dict(raw)
    o30 = raw.get("orders_last_30d", 0)
    o90 = raw.get("orders_last_90d", 1)
    feat["engagement_decay"] = o30 / (o90 + 1e-6)
    aov = raw.get("avg_order_value", 0)
    freq = raw.get("order_frequency", 0)
    feat["order_value_frequency"] = aov * freq
    rating = raw.get("avg_user_rating", 3.0)
    delay = raw.get("avg_delivery_delay_min", 0)
    complaints = raw.get("num_complaints", 0)
    feat["experience_score"] = rating - 0.1 * delay - 0.3 * complaints
    recency = raw.get("days_since_last_order", 0)
    feat["recency_frequency"] = recency / (freq + 1e-6)
    tenure = raw.get("account_age_days", 1)
    feat["complaint_rate"] = complaints / (tenure + 1e-6) * 365
    return feat


def _classify_risk(prob: float) -> tuple:
    if prob < 0.30:
        return "low", "No action needed -- user is healthy."
    elif prob < 0.60:
        return "medium", "Send loyalty reward or push notification to re-engage."
    else:
        return "high", "Offer personalised discount and schedule direct outreach."


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------
class ChurnRequest(BaseModel):
    orders_last_30d: float
    orders_last_90d: float
    avg_order_value: float
    days_since_last_order: float
    order_frequency: float
    cancellation_rate: float
    avg_delivery_delay_min: float
    avg_user_rating: float
    num_complaints: float
    discount_usage_rate: float
    app_sessions_per_week: float
    preferred_order_hour: float
    account_age_days: float


class ChurnResponse(BaseModel):
    churn_probability: float
    risk_level: str
    recommended_action: str
    features_used: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post("/predict", response_model=ChurnResponse)
async def predict_churn(req: ChurnRequest):
    """Predict churn probability for a user."""
    if not _initialized:
        raise HTTPException(503, f"Churn service unavailable: {_error}")

    raw = req.model_dump()
    full = _engineer(raw)

    row = {"user_id": 0}
    row.update({f: full.get(f, 0.0) for f in NUMERIC_RAW})
    row["churn"] = 0
    row.update({f: full.get(f, 0.0) for f in ENGINEERED})
    df_input = pd.DataFrame([row])

    X = _transformer.transform(df_input)
    prob = float(_model.predict_proba(X)[:, 1][0])
    risk_level, action = _classify_risk(prob)

    return ChurnResponse(
        churn_probability=round(prob, 4),
        risk_level=risk_level,
        recommended_action=action,
        features_used=len(ALL_FEATURES),
    )


@router.get("/health")
async def health():
    return {
        "status": "ok" if _initialized else "unavailable",
        "service": "churn-prediction",
        "error": _error,
    }
