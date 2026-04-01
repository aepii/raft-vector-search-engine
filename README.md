# Distributed Vector Search Engine

A *distributed system* for storing and querying [vector embeddings](https://www.pinecone.io/learn/vector-database/). This project implements a partitioned vector index utilizing a coordinator-worker architecture.

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/aepii/vector-search-engine.git
cd vector-search-engine
git clone https://github.com/aepii/vector-search-engine.git
cd vector-search-engine
```

### 2. Setup a virtual environment

```bash
# Create the environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on macOS/Linux
source .venv/bin/activate
```

### 3.  Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Compile the Protobuf Definitions

```bash
python -m grpc_tools.protoc \
    -I protos \
    --python_out=src \
    --grpc_python_out=src \
    protos/vector_store.proto
```

### 5. Configure Environment

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

## Running Locally

Start each component in separate terminals.

### 1. Navigate to the source directory
```bash
cd src
```

### 2. Start the Coordinator
```bash
python -m coordinator
```

### 3. Start Server Nodes (Shards)

Run one server per terminal:
```bash
$env:SERVER_HOST=50051; python -m server
```
```bash
$env:SERVER_HOST=50052; python -m server
```
```bash
$env:SERVER_HOST=50053; python -m server
```

### 4. Run Benchmark (after all services are ready)
```bash
python -m benchmarks.benchmark
```

