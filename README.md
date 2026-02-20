# Distributed Crash-Fault-Tolerant Vector Search Engine

A *high-availability distributed system* for storing and querying [vector embeddings](https://www.pinecone.io/learn/vector-database/). This project implements a consistent, replicated vector index using the [**Raft Consensus Algorithm**](https://raft.github.io/raft.pdf).

## Setup and Installation
### 1. Clone the repository:

```bash
git clone https://github.com/aepii/raft-vector-search-engine.git
cd raft-vector-search-engine
```

### 2. Setup a virtual environment:

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

## Execution
To run the demo script and verify the vector search:

```bash
python src/main.py
```