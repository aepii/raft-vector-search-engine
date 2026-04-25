FROM python:3.13-slim

WORKDIR /app

COPY requirements-shard.txt .
RUN pip install --no-cache-dir -r requirements-shard.txt

COPY protos/ protos/
COPY src/ src/

WORKDIR /app/src

CMD ["python", "-m", "server"]
