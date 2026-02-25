# Bad Review Predictor API

Pre-shipment risk scoring for marketplace orders.

Predicts whether an order will receive a ≤2 star review **before it ships**, enabling proactive logistics intervention.

---

## 📊 Business Case

| Metric | Value |
|--------|--------|
| AUC-ROC | 0.63 |
| Decision threshold | 0.487 |
| Flag rate | ~30% of orders |
| Recall on bad reviews | 43% |
| Training set | 109,370 delivered orders |
| Date range | Sep 2016 – Aug 2018 |

### What this means operationally

Out of every 100 orders, the model flags ~30 as risky.

Of all orders that *would* generate a bad review, **43% are caught before dispatch**, giving operations teams a window to:

- Tighten seller SLAs  
- Send proactive delivery ETAs  
- Reroute through faster carriers  
- Improve customer communication  

---

## 📦 Data

Raw datasets are not included in the repo (GitHub size limits).

Download from:  
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

### Required files:

- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_order_reviews_dataset.csv`
- `olist_sellers_dataset.csv`
- `olist_customers_dataset.csv`
- `olist_geolocation_dataset.csv`

---

## 📁 Repo Structure

```
.
├── serve.py                      # FastAPI app + SHAP explainer
├── bad_review_prod_model.txt     # Trained LightGBM model (17 trees)
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Production container
├── README.md
├── .gitignore
├── notebooks/
│   └── modeling.ipynb            # Full training pipeline
├── experiments/
│   ├── feature_importance.png
│   ├── gate1_delivery_signal.png
│   ├── gate2b_class_balance.png
│   └── gate3b_cold_start.png
└── ai_chat_logs/
    └── perplexity_chat.txt       # Development discussion log
```

---

## 🐳 Run with Docker

### Build

```bash
docker build -t bad-review-api .
```

### Run

```bash
docker run -p 8000:8000 bad-review-api
```

### Verify

```bash
curl http://localhost:8000/health
```

---

## 🔮 Predict

### Endpoint

```
POST /predict
```

### Required Fields

| Field | Type | Description |
|--------|------|-------------|
| `freight_value` | float | Shipping cost in BRL |
| `geo_distance_km` | float | Seller-to-customer distance |
| `seller_hist_bad_review_rate_10` | float | Seller's recent bad review rate (0–1) |
| `expected_window_days` | float | Estimated delivery window (days) |
| `seller_hist_delay_median_10` | float | Seller's median delivery delay (days) |

---

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "freight_value": 35.2,
    "geo_distance_km": 1250,
    "seller_hist_bad_review_rate_10": 0.42,
    "expected_window_days": 18,
    "seller_hist_delay_median_10": 4.2
  }'
```

---

### Example Response

```json
{
  "risk_score": 0.72,
  "risky": true,
  "threshold": 0.487,
  "shap_top3": [
    {
      "feature": "geo_distance_km",
      "shap_value": 0.23,
      "feature_value": 1250
    },
    {
      "feature": "seller_hist_bad_review_rate_10",
      "shap_value": 0.18,
      "feature_value": 0.42
    },
    {
      "feature": "freight_value",
      "shap_value": 0.12,
      "feature_value": 35.2
    }
  ]
}
```

---

## 🧭 Other Endpoints

| Route | Method | Description |
|--------|--------|-------------|
| `/health` | GET | Liveness check + model stats |
| `/features` | GET | Full feature list (36 features) |
| `/example` | GET | Copy-paste curl example |
| `/docs` | GET | Swagger UI (auto-generated) |

---

## 🛠 Tech Stack

- **Model:** LightGBM 4.5 binary classifier  
- **Explainability:** SHAP TreeExplainer (top-3 per prediction)  
- **API:** FastAPI + Pydantic v2 validation  
- **Container:** Python 3.11-slim Docker image  

---

## ✅ Build Checklist

```bash
# Verify files before building:
ls -1

# Should include:
# bad_review_prod_model.txt
# serve.py
# requirements.txt
# Dockerfile
# README.md

docker build -t bad-review-api .
docker run -p 8000:8000 bad-review-api

# Swagger UI auto-docs:
open http://localhost:8000/docs
```

---

## 🧪 Quick Test

Once the server is running, test with a high-risk scenario:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "freight_value": 80.0,
    "geo_distance_km": 3000,
    "seller_hist_bad_review_rate_10": 0.8,
    "expected_window_days": 30,
    "seller_hist_delay_median_10": 10.0
  }'


---


## 📄 License

MIT