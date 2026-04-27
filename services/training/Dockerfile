FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    boto3 \
    onnx \
    onnxruntime \
    psycopg2-binary \
    numpy

COPY train_gismo.py /app/train_gismo.py

ENTRYPOINT ["python", "train_gismo.py"]
