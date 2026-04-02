"""
Shard distribution check.

Requires all three shards to be running and seeded (e.g. after a benchmark run).
Run from src/:
    pytest tests/test_shard_distribution.py -s
"""
import grpc
import pytest
import vector_store_pb2
import vector_store_pb2_grpc

SHARD_HOSTS = ["localhost:50051", "localhost:50052", "localhost:50053"]
# With 150 virtual nodes and 3 shards the distribution should be ~33% each.
# Allow ±10% of total (i.e. 23%–43%) to avoid flakiness.
MIN_FRACTION = 0.23
MAX_FRACTION = 0.43


@pytest.fixture(scope="module")
def shard_counts():
    counts = {}
    for host in SHARD_HOSTS:
        stub = vector_store_pb2_grpc.VectorStoreStub(grpc.insecure_channel(host))
        resp = stub.Count(vector_store_pb2.CountRequest())
        counts[host] = resp.count
    return counts


def test_each_shard_has_items(shard_counts):
    for host, count in shard_counts.items():
        assert count > 0, f"Shard {host} has 0 items — routing may be broken"


def test_distribution_is_roughly_even(shard_counts):
    total = sum(shard_counts.values())
    for host, count in shard_counts.items():
        fraction = count / total
        assert MIN_FRACTION <= fraction <= MAX_FRACTION, (
            f"Shard {host} holds {fraction:.1%} of items "
            f"(expected {MIN_FRACTION:.0%}–{MAX_FRACTION:.0%}). "
            f"Per-shard: {shard_counts}"
        )
