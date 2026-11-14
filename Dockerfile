
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-api.txt .

RUN python -m pip install --no-cache-dir -r requirements-api.txt

COPY api/ ./api/
COPY src/ ./src/
COPY models/checkpoints/P-1_best.pt ./models/checkpoints/P-1_best.pt
COPY data/processed/scaler.joblib ./data/processed/scaler.joblib

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

