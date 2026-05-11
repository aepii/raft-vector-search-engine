#!/usr/bin/env bash
# Run from the repo root after any change to protos/vector_store.proto.
# Regenerates src/vector_store_pb2.py and src/vector_store_pb2_grpc.py.
set -euo pipefail

python -m grpc_tools.protoc \
    -I protos \
    --python_out=src \
    --grpc_python_out=src \
    protos/vector_store.proto

echo "Proto stubs regenerated in src/"
