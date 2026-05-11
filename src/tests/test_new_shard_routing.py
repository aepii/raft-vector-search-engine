"""
Verify that registering a new shard causes new upserts to route to it.

Requires:
  - coordinator running on port 50050
  - shards on 50051, 50052, 50053 (existing)
  - a 4th shard running on port 50054

Run from src/:
    pytest tests/test_new_shard_routing.py -v -s
"""
import grpc
import pytest
import hashlib
import vector_store_pb2
import vector_store_pb2_grpc

COORDINATOR = "localhost:50050"
NEW_SHARD = "localhost:50054"

# Items whose IDs hash to the new shard — we upsert enough that at least
# some are guaranteed to land there after the ring is rebalanced (~25% of keys).
NUM_ITEMS = 200


def make_id(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**63 - 1)


ITEMS = [(make_id(f"routing-test-item-{i}"), f"routing test item {i}") for i in range(NUM_ITEMS)]


@pytest.fixture(scope="module")
def ctrl():
    return vector_store_pb2_grpc.CoordinatorControlStub(grpc.insecure_channel(COORDINATOR))


@pytest.fixture(scope="module")
def coordinator():
    return vector_store_pb2_grpc.VectorStoreStub(grpc.insecure_channel(COORDINATOR))


@pytest.fixture(scope="module")
def new_shard():
    return vector_store_pb2_grpc.VectorStoreStub(grpc.insecure_channel(NEW_SHARD))


@pytest.fixture(scope="module", autouse=True)
def register_and_cleanup(ctrl):
    resp = ctrl.RegisterNode(vector_store_pb2.NodeRequest(host=NEW_SHARD))
    assert resp.success, f"Failed to register new shard: {resp.message}"
    yield
    ctrl.DeregisterNode(vector_store_pb2.NodeRequest(host=NEW_SHARD))


def test_new_shard_starts_empty(new_shard):
    count = new_shard.Count(vector_store_pb2.CountRequest()).count
    assert count == 0, f"Expected new shard to start empty, got {count} items"


def test_new_shard_receives_writes(coordinator, new_shard):
    request_items = [vector_store_pb2.UpsertItem(id=item_id, text=text) for item_id, text in ITEMS]
    coordinator.UpsertBatch(vector_store_pb2.UpsertBatchRequest(items=request_items, trace_id="test-routing"))

    count = new_shard.Count(vector_store_pb2.CountRequest()).count
    assert count > 0, (
        f"New shard received 0 items after upserting {NUM_ITEMS} items. "
        "Ring routing may not be directing writes to the new node."
    )
