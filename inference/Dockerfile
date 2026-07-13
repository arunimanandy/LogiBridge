FROM python:3.11-slim

WORKDIR /app

COPY requirements_inference.txt .
RUN pip install --no-cache-dir -r requirements_inference.txt

COPY inference_service.py .
COPY training_stats.npy .
COPY model.tflite .

ENV MODEL_PATH=/app/model.tflite
ENV STATS_PATH=/app/training_stats.npy
ENV TRUCK_ID=TRUCK_001
ENV MQTT_HOST=host.docker.internal
ENV MQTT_PORT=1883

CMD ["python", "inference_service.py"]
