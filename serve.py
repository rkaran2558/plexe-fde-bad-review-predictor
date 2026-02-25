from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import lightgbm as lgb
import numpy as np
import shap
from typing import List

# ── Model & Explainer (loaded once at startup) ────────────────────────────────
MODEL_PATH = "bad_review_prod_model.txt"
THRESHOLD = 0.487

model = lgb.Booster(model_file=MODEL_PATH)
explainer = shap.TreeExplainer(model)

# All 36 features in exact order from the model file
FEATURE_NAMES = [
    "price", "freight_value", "product_name_lenght", "product_description_lenght",
    "product_photos_qty", "product_weight_g", "product_length_cm", "product_height_cm",
    "product_width_cm", "seller_zip_code_prefix", "customer_zip_code_prefix",
    "payment_installments", "orig_idx", "seller_prior_orders", "seller_age_days",
    "seller_n_orders_30d", "seller_n_orders_90d", "seller_hist_late_rate_10",
    "seller_hist_bad_review_rate_10", "seller_hist_delay_median_10",
    "seller_lat", "seller_lng", "customer_lat", "customer_lng",
    "geo_distance_km", "cross_state", "expected_window_days", "freight_price_ratio",
    "purchase_dow", "purchase_month", "purchase_hour", "product_volume_cm3",
    "has_seller_history", "product_category_name_encoded",
    "seller_state_encoded", "customer_state_encoded"
]

app = FastAPI(
    title="Bad Review Predictor API",
    description="Pre-shipment risk scoring for marketplace orders. Predicts ≤2 star reviews before dispatch.",
    version="1.0.0"
)


# ── Request / Response Schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    freight_value: float = Field(..., description="Shipping cost in BRL")
    geo_distance_km: float = Field(..., description="Seller-to-customer distance in km")
    seller_hist_bad_review_rate_10: float = Field(..., ge=0, le=1, description="Seller's recent bad review rate (0–1)")
    expected_window_days: float = Field(..., description="Estimated delivery window in days")
    seller_hist_delay_median_10: float = Field(..., description="Seller's median delivery delay in days")


class SHAPFeature(BaseModel):
    feature: str
    shap_value: float
    feature_value: float


class PredictResponse(BaseModel):
    risk_score: float
    risky: bool
    threshold: float
    shap_top3: List[SHAPFeature]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "threshold": THRESHOLD,
        "num_trees": model.num_trees(),
        "num_features": len(FEATURE_NAMES)
    }


@app.get("/features")
def features():
    return {"features": FEATURE_NAMES, "count": len(FEATURE_NAMES)}


@app.get("/example")
def example():
    return {
        "curl": (
            'curl -X POST "http://localhost:8000/predict" '
            '-H "Content-Type: application/json" '
            '-d \'{"freight_value": 35.2, "geo_distance_km": 1250, '
            '"seller_hist_bad_review_rate_10": 0.42, '
            '"expected_window_days": 18, "seller_hist_delay_median_10": 4.2}\''
        )
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Build full feature vector (36 features), zeros for unobserved at inference time
    row = np.zeros((1, len(FEATURE_NAMES)))
    row[0, FEATURE_NAMES.index("freight_value")]                 = req.freight_value
    row[0, FEATURE_NAMES.index("geo_distance_km")]               = req.geo_distance_km
    row[0, FEATURE_NAMES.index("seller_hist_bad_review_rate_10")]= req.seller_hist_bad_review_rate_10
    row[0, FEATURE_NAMES.index("expected_window_days")]          = req.expected_window_days
    row[0, FEATURE_NAMES.index("seller_hist_delay_median_10")]   = req.seller_hist_delay_median_10

    # Predict
    score = float(model.predict(row)[0])
    risky = score >= THRESHOLD

    # SHAP top-3
    shap_values = explainer.shap_values(row)[0]
    top3_idx = np.argsort(np.abs(shap_values))[::-1][:3]
    shap_top3 = [
        SHAPFeature(
            feature=FEATURE_NAMES[i],
            shap_value=round(float(shap_values[i]), 4),
            feature_value=round(float(row[0, i]), 4)
        )
        for i in top3_idx
    ]

    return PredictResponse(
        risk_score=round(score, 4),
        risky=risky,
        threshold=THRESHOLD,
        shap_top3=shap_top3
    )
