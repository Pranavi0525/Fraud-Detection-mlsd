FROM python:3.11-slim

WORKDIR /app

COPY fraud_detection_service/requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn joblib xgboost scikit-learn pandas numpy

COPY fraud_detection_service/ ./fraud_detection_service/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "fraud_detection_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
