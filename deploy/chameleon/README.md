# Chameleon Cloud Deployment

## Node layout

| Node | Role | Script |
| ---- | ---- | ------ |
| node-0 | Coordinator | `start-coordinator.sh` |
| node-1 | Shard 0 | `start-shard.sh` |
| node-2 | Shard 1 | `start-shard.sh` |
| node-3 | Shard 2 | `start-shard.sh` |
| node-4 (optional) | Benchmark | `start-benchmark.sh` |

## Steps

### 1. Lease nodes and get IPs

Submit a lease for at least 4 nodes on Chameleon. Once active, note the external IP of each node.

### 2. Install Docker on each node

```bash
curl -fsSL https://get.docker.com | sh
```

### 3. Pull the images on each node

On the coordinator node:

```bash
docker pull ghcr.io/aepii/vector-search-engine-coordinator:latest
```

On each shard node:

```bash
docker pull ghcr.io/aepii/vector-search-engine-shard:latest
```

### 4. Start the coordinator

SSH into node-0. Edit `start-coordinator.sh` and replace `<COORDINATOR_IP>` with node-0's external IP, then run:

```bash
bash start-coordinator.sh
```

### 5. Start the shards

SSH into each shard node. Edit `start-shard.sh` and replace:

- `<COORDINATOR_IP>` with node-0's external IP
- `<SHARD_IP>` with this node's own external IP

Then run:

```bash
bash start-shard.sh
```

Startup order does not matter — shards retry heartbeats until the coordinator responds.

### 6. Verify

Check that all shards have registered by looking at the coordinator logs:

```bash
docker logs coordinator
```

You should see three `registered via heartbeat` lines.

### 7. Run the benchmark

On the benchmark node (or any node that can reach the coordinator), edit `start-benchmark.sh` with the coordinator IP and run:

```bash
bash start-benchmark.sh
```

## Stopping

```bash
docker stop coordinator   # on coordinator node
docker stop shard         # on each shard node
```

To wipe shard data:

```bash
docker stop shard && docker rm shard && docker volume rm shard-data
```
