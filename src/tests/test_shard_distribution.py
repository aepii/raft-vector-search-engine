"""
Shard distribution check.

Requires the docker compose cluster to be running and responsive.
Run: docker compose up -d, then pytest -m integration

Run from src/:
    pytest tests/test_shard_distribution.py -v
"""
import grpc
import pytest
import vector_store_pb2
import vector_store_pb2_grpc

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("compose_cluster")]

COORDINATOR = "localhost:50050"
SHARD_HOSTS = ["localhost:50051", "localhost:50052", "localhost:50053", "localhost:50054", "localhost:50055"]
# With RF=3 and 5 shards each item lands on 3 shards, so total copies = items * 3.
# Each shard's share of that total is ~20%; allow ±8% for hash ring skew.
MIN_FRACTION = 0.12
MAX_FRACTION = 0.28
SEED_IDS_START = 9000
SEED_COUNT = 60


@pytest.fixture(scope="module", autouse=True)
def seed_data(compose_cluster):
    """Upsert a small fixed dataset so shards have items before distribution checks."""
    ch = grpc.insecure_channel(COORDINATOR)
    stub = vector_store_pb2_grpc.VectorStoreStub(ch)
    items = [
        vector_store_pb2.UpsertItem(id=SEED_IDS_START + i, text=f"distribution seed item {i}")
        for i in range(SEED_COUNT)
    ]
    stub.UpsertBatch(
        vector_store_pb2.UpsertBatchRequest(items=items, trace_id="dist-seed")
    )
    ch.close()


@pytest.fixture(scope="module")
def shard_counts(seed_data):
    counts = {}
    for host in SHARD_HOSTS:
        stub = vector_store_pb2_grpc.VectorStoreStub(grpc.insecure_channel(host))
        counts[host] = stub.Count(vector_store_pb2.CountRequest()).count
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
