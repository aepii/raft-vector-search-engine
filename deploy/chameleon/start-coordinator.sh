#!/usr/bin/env bash
# Run on the coordinator node.
# Replace COORDINATOR_IP with this node's external Chameleon IP.
set -euo pipefail

# COORDINATOR_IP="<COORDINATOR_IP>" # not needed unless/until we do raft

docker run -d \
    --name coordinator \
    --restart unless-stopped \
    -p 50050:50050 \
    -e COORDINATOR_PORT=50050 \
    -e REPLICATION_FACTOR=0 \
    ghcr.io/aepii/vector-search-engine-coordinator:latest
