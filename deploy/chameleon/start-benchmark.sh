#!/usr/bin/env bash
# Run on the benchmark node (or any node that can reach the coordinator).
# Replace COORDINATOR_IP with the coordinator node's external Chameleon IP.
set -euo pipefail

COORDINATOR_IP="<COORDINATOR_IP>"

docker run --rm \
    -e COORDINATOR_HOST="${COORDINATOR_IP}:50050" \
    ghcr.io/aepii/vector-search-engine-coordinator:latest \
    python -m benchmarks.benchmark
