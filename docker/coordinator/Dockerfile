FROM python:3.13-slim

WORKDIR /app

COPY requirements-coordinator.txt .
RUN pip install --no-cache-dir -r requirements-coordinator.txt

COPY protos/ protos/
COPY src/ src/

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

WORKDIR /app/src

CMD ["python", "-m", "coordinator"]
