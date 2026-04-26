FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gismo_model.py .
COPY data_loader.py .
COPY train.py .
COPY feedback_train.py .
COPY config.yaml .

EXPOSE 8001
CMD ["python", "feedback_train.py"]
