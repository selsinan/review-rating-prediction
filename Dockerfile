FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir uv
RUN uv pip install --system --strict pyproject.toml

# Install additional packages with specific versions to match training
RUN pip install --no-cache-dir \
    uvicorn \
    fastapi \
    mlflow \
    scikit-learn==1.5.0 \
    pandas \
    numpy

COPY src/ ./src/
COPY configs/ ./configs/

EXPOSE 8000

# Set environment variables for MLflow
ENV PYTHONPATH=/app/src
ENV MLFLOW_TRACKING_URI=file:///mlruns
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlruns

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]