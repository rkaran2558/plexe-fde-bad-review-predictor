FROM python:3.11-slim

WORKDIR /app

# System deps for LightGBM + SHAP
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    gcc \
    curl \ 
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serve.py .
COPY bad_review_prod_model.txt .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
