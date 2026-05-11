"""
Tests for VectorStoreClient behavior and contract.

These tests run against the client directly — no live gRPC server required.

Run from src/:
    pytest tests/test_client.py -v
"""
import pytest
from unittest.mock import MagicMock, patch
import vector_store_pb2


def test_default_host_points_to_coordinator():
    """VectorStoreClient's default host should be the coordinator (port 50050),
    not a shard (port 50051). Currently hardcoded to localhost:50051."""
    import client.vector_store_client as m

    assert m.COORDINATOR_HOST == "localhost:50050", (
        f"COORDINATOR_HOST defaults to {m.COORDINATOR_HOST!r} — "
        "should be localhost:50050 (coordinator), not a shard port"
    )


def test_search_returns_tuples():
    """search() should return list[tuple[str, float]]. The return type annotation
    incorrectly says list[str], and test_search.py:32 asserts results[0] against
    a bare string — which always fails because results[0] is a (text, score) tuple."""
    fake_response = vector_store_pb2.SearchResponse(
        results=[vector_store_pb2.SearchResult(text="Python is great", score=0.95)]
    )

    with patch("client.vector_store_client.grpc.insecure_channel"):
        with patch("client.vector_store_client.vector_store_pb2_grpc.VectorStoreStub") as MockStub:
            MockStub.return_value.Search.return_value = fake_response
            from client.vector_store_client import VectorStoreClient
            c = VectorStoreClient(host="localhost:50050")
            results = c.search("Python", top_k=1)

    assert isinstance(results[0], tuple)
    text, score = results[0]
    assert text == "Python is great"
    assert isinstance(score, float)
