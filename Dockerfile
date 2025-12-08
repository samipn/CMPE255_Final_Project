# Dockerfile for Mental Health Psychometric Inference Service
# CMPE 255 Final Project

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY app/ ./app/

# Copy trained model (if available)
COPY trained_model/ ./trained_model/ 2>/dev/null || true

# Expose port
EXPOSE 8080

# Set environment variables
ENV MODEL_DIR=/app/trained_model
ENV MODEL_NAME=xlm-roberta-large
ENV PYTHONUNBUFFERED=1

# Run the inference server
CMD ["uvicorn", "src.inference:app", "--host", "0.0.0.0", "--port", "8080"]
