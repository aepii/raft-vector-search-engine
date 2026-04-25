# Distributed Vector Search Engine

A *distributed system* for storing and querying [vector embeddings](https://www.pinecone.io/learn/vector-database/). This project implements a partitioned vector index utilizing a coordinator-worker architecture.

## Architecture

```text
Client
  └─► Coordinator (port 50050)  — hash-routes writes, fans out reads
          ├─► Shard 0 (port 50051)
          ├─► Shard 1 (port 50052)
          └─► Shard 2 (port 50053)
```

Writes (`Upsert`, `UpsertBatch`) are routed via a consistent hash ring (SHA-256, 150 virtual nodes per host) to the next N clockwise physical nodes, where N is configurable via `REPLICATION_FACTOR` (default `0` = all nodes). At small scale this gives full replication — every shard holds every vector. Reads (`Search`) broadcast to all shards in parallel, merge results by score, and deduplicate before returning.

Shards self-register by sending periodic heartbeats to the coordinator. The coordinator tracks liveness via a background sweep and automatically removes shards that stop responding. Startup order does not matter — shards can come up before or after the coordinator and will register on their first successful heartbeat.

When a shard restarts, it seeds itself from healthy peers before registering — a recovering shard never enters the routing table with a stale or empty database. On startup the shard calls `GetPeers` on the coordinator, which divides the SHA-256 hash space into equal arcs and assigns one arc per donor. The shard opens parallel `Dump` streams to each donor, pulls its slice of the dataset, and writes items into its local DB; only after all streams complete does the heartbeat loop start. Set `SKIP_STATE_TRANSFER=true` to bypass this for local dev.

The coordinator also exposes a `CoordinatorControl` service on port 50050 for manual registration and deregistration if needed:

```bash
# Example: manually register or deregister a shard (via grpcurl or a client script)
RegisterNode   { host: "localhost:50054" }
DeregisterNode { host: "localhost:50054" }
```

## Operational constraints

- **State transfer window.** Writes arriving on other shards *during* a dump are missed by the recovering shard. The gap is bounded by dump duration; the coordinator logs a warning on re-registration. This is acceptable for the current scale but the write-buffer approach is the natural follow-on.
- **Permanent node loss under partial replication.** Ring positions are derived from the host string. Under the current default (full replication, N = total nodes) any replacement hostname works — state transfer fills it completely. Under partial replication (N < total nodes), a replacement must reuse the dead node's hostname to land in the same ring positions and restore full N-way replication. A different hostname occupies different positions, leaving some keys underreplicated until a redistribution pass is run. No redistribution mechanism is currently implemented.
- **Coordinator is a single point of failure.** If the coordinator crashes, the system stops routing. Raft consensus across coordinator replicas is the correct fix; see the project roadmap.

## Setup and Installation

### 1. Clone the repository

```bash
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

**All subsequent commands assume the virtual environment is active**.

### 3.  Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the pre-commit hook

`.git` is not tracked by git, so you have to run the following for the hook to be active in your dev environment.

```bash
ln -s ../../scripts/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

This blocks commits where `protos/vector_store.proto` and the compiled stubs are out of sync. If the proto changes, regenerate with:

```bash
bash scripts/generate_protos.sh
```

The detection/blocking hook is automatic, but *regeneration is still manual*.

### 5. Configure Environment

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

*Currently* there are no secrets in here, but `.env` is still `.gitignore`d. We don't anticipate needing any secrets here.

## Running Locally

### With Docker Compose (recommended)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine on Linux). Make sure it's running first, Otherwise you'll get an `unable to get image` error.

```bash
docker compose up --build
```

This starts the coordinator and three shards on a shared bridge network with named volumes for shard persistence. The coordinator image is large (~1.7 GB) due to the embedded ML model; the first build takes several minutes.

To run the benchmark against the running cluster:

```bash
docker compose run --rm coordinator python -m benchmarks.benchmark
```

To stop and remove containers (volumes are preserved):

```bash
docker compose down
```

To also wipe shard data:

```bash
docker compose down -v
```

### Without Docker (bare processes)

Start each component in separate terminals from within `src/`.

```bash
cd src
```

```bash
python -m coordinator
```

```bash
$env:SERVER_PORT=50051; python -m server
$env:SERVER_PORT=50052; python -m server
$env:SERVER_PORT=50053; python -m server
```

```bash
python -m benchmarks.benchmark
```

## Deploying on Chameleon Cloud

> These instructions are a starting point. Once we establish our actual Chameleon workflow we can update this.

Deployment scripts are in `deploy/chameleon/`. The expected node layout is one VM per role:

| Node | Role | Script |
| ---- | ---- | ------ |
| node-0 | Coordinator | `start-coordinator.sh` |
| node-1 | Shard 0 | `start-shard.sh` |
| node-2 | Shard 1 | `start-shard.sh` |
| node-3 | Shard 2 | `start-shard.sh` |

See [`deploy/chameleon/README.md`](deploy/chameleon/README.md) for more detail.

Docker images are published to GHCR automatically on every push to `main` via GitHub Actions. Each node pulls its image directly without having to build it itself.
