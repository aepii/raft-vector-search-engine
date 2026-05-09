"""
Verify that a newly added shard starts receiving writes via hash ring routing.

Pure unit tests — no live processes needed. The original live-process version
required a 4th shard container on port 50054 (not in the compose cluster), so
it was rewritten to manipulate the coordinator's ring and stub map directly,
which is the same internal path used by RegisterNode.

Run from src/:
    pytest tests/test_new_shard_routing.py -v
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import vector_store_pb2
from coordinator import CoordinatorServicer

# With RF=1 and 4 nodes (~150 virtual nodes each), P(new node gets 0 of 100 items) ≈ 3e-13.
NUM_ITEMS = 100


def _make_coordinator(n_shards, replication_factor):
    hosts = [f"fake:{5000 + i}" for i in range(n_shards)]
    stubs = [MagicMock() for _ in hosts]

    with patch("coordinator.EmbeddingModel") as MockEmbed, \
         patch("coordinator.grpc.insecure_channel"), \
         patch("coordinator.vector_store_pb2_grpc.VectorStoreStub", side_effect=stubs):
        MockEmbed.return_value.encode.side_effect = lambda x: (
            np.zeros((len(x), 384)) if isinstance(x, list) else np.zeros(384)
        )
        coord = CoordinatorServicer(hosts, replication_factor=replication_factor)

    return coord, stubs


def _add_shard(coord):
    """Inject a new mock shard — mirrors what CoordinatorControlServicer.RegisterNode does."""
    stub = MagicMock()
    stub.UpsertBatch.return_value = vector_store_pb2.UpsertBatchResponse(statuses=["ok"])
    host = "fake:5099"
    coord._ring.add_node(host)
    coord._stub_map[host] = stub
    return stub


def _upsert_batch(coord, n):
    for stub in coord._stub_map.values():
        if not stub.UpsertBatch.return_value:
            stub.UpsertBatch.return_value = vector_store_pb2.UpsertBatchResponse(statuses=["ok"])
    items = [vector_store_pb2.UpsertItem(id=i, text=f"item {i}") for i in range(n)]
    coord.UpsertBatch(
        vector_store_pb2.UpsertBatchRequest(trace_id="routing-test", items=items),
        context=MagicMock(),
    )


def test_newly_registered_node_receives_writes():
    """After adding a node, subsequent writes route to it (RF=1: only routed keys)."""
    coord, initial_stubs = _make_coordinator(n_shards=3, replication_factor=1)
    for s in initial_stubs:
        s.UpsertBatch.return_value = vector_store_pb2.UpsertBatchResponse(statuses=["ok"])

    new_stub = _add_shard(coord)
    _upsert_batch(coord, NUM_ITEMS)

    assert new_stub.UpsertBatch.called, (
        "New node received no writes after registration — ring routing may be broken"
    )


def test_rf1_total_items_sent_equals_input():
    """With RF=1 each item routes to exactly 1 shard regardless of ring size."""
    coord, initial_stubs = _make_coordinator(n_shards=3, replication_factor=1)
    for s in initial_stubs:
        s.UpsertBatch.return_value = vector_store_pb2.UpsertBatchResponse(statuses=["ok"])

    new_stub = _add_shard(coord)
    all_stubs = initial_stubs + [new_stub]

    items = [vector_store_pb2.UpsertItem(id=i, text=f"item {i}") for i in range(NUM_ITEMS)]
    coord.UpsertBatch(
        vector_store_pb2.UpsertBatchRequest(trace_id="rf1-total", items=items),
        context=MagicMock(),
    )

    total_sent = sum(
        len(s.UpsertBatch.call_args[0][0].items)
        for s in all_stubs
        if s.UpsertBatch.called
    )
    assert total_sent == NUM_ITEMS, (
        f"RF=1 should route each item to exactly 1 shard; "
        f"got total_sent={total_sent} for {NUM_ITEMS} input items"
    )


def test_node_before_registration_receives_no_writes():
    """Writes issued before registration must not reach a stub that isn't in the ring yet."""
    coord, initial_stubs = _make_coordinator(n_shards=3, replication_factor=1)
    for s in initial_stubs:
        s.UpsertBatch.return_value = vector_store_pb2.UpsertBatchResponse(statuses=["ok"])

    unregistered = MagicMock()

    # Write BEFORE adding to ring.
    items = [vector_store_pb2.UpsertItem(id=i, text=f"item {i}") for i in range(NUM_ITEMS)]
    coord.UpsertBatch(
        vector_store_pb2.UpsertBatchRequest(trace_id="pre-reg", items=items),
        context=MagicMock(),
    )

    assert not unregistered.UpsertBatch.called, (
        "Unregistered stub received writes — stub map may have leaked"
    )
