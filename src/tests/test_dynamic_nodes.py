"""
RegisterNode / DeregisterNode integration tests.

Requires the docker compose cluster (coordinator + shards) to be running.
Run: docker compose up -d, then pytest -m integration

Run from src/:
    pytest tests/test_dynamic_nodes.py -v
"""
import grpc
import pytest
import vector_store_pb2
import vector_store_pb2_grpc

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("compose_cluster")]

COORDINATOR = "localhost:50050"
EXTRA_HOST = "localhost:59998"
INITIAL_NODE_COUNT = 5


@pytest.fixture(scope="module")
def ctrl():
    stub = vector_store_pb2_grpc.CoordinatorControlStub(grpc.insecure_channel(COORDINATOR))
    yield stub
    # Ensure the extra node is removed even if a test fails mid-way.
    stub.DeregisterNode(vector_store_pb2.NodeRequest(host=EXTRA_HOST))


def test_register_new_node(ctrl):
    resp = ctrl.RegisterNode(vector_store_pb2.NodeRequest(host=EXTRA_HOST))
    assert resp.success, f"RegisterNode failed: {resp.message}"
    assert resp.node_count == INITIAL_NODE_COUNT + 1, (
        f"Expected {INITIAL_NODE_COUNT + 1} nodes, got {resp.node_count}"
    )


def test_register_same_node_is_idempotent(ctrl):
    resp = ctrl.RegisterNode(vector_store_pb2.NodeRequest(host=EXTRA_HOST))
    assert resp.success, f"Re-register should still succeed: {resp.message}"
    assert resp.node_count == INITIAL_NODE_COUNT + 1, (
        f"Idempotent re-register should not change node count, got {resp.node_count}"
    )


def test_deregister_node(ctrl):
    resp = ctrl.DeregisterNode(vector_store_pb2.NodeRequest(host=EXTRA_HOST))
    assert resp.success, f"DeregisterNode failed: {resp.message}"
    assert resp.node_count == INITIAL_NODE_COUNT, (
        f"Expected {INITIAL_NODE_COUNT} nodes after deregister, got {resp.node_count}"
    )


def test_deregister_unknown_node_fails(ctrl):
    resp = ctrl.DeregisterNode(vector_store_pb2.NodeRequest(host=EXTRA_HOST))
    assert not resp.success, "Deregistering a node not in the ring should return success=False"
