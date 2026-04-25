#!/usr/bin/env bash
# Run on each shard node.
# Replace COORDINATOR_IP with the coordinator node's external Chameleon IP.
# Replace SHARD_IP with this node's own external Chameleon IP.
set -euo pipefail

COORDINATOR_IP="<COORDINATOR_IP>"
SHARD_IP="<SHARD_IP>"

docker run -d \
    --name shard \
    --restart unless-stopped \
    -p 50051:50051 \
    -v shard-data:/data \
    -e SERVER_PORT=50051 \
    -e COORDINATOR_HOST="${COORDINATOR_IP}:50050" \
    -e SHARD_HOST="${SHARD_IP}:50051" \
    -e DB_PATH=/data/shard.db \
    ghcr.io/aepii/vector-search-engine-shard:latest
