"""
Tests for shard (server.py) handler behavior.

These tests run against the shard servicer directly — no live gRPC server required.

Run from src/:
    pytest tests/test_shard.py -v
"""
import pytest
from unittest.mock import MagicMock, patch
import vector_store_pb2


def test_upsert_handler_does_not_crash():
    """Shard Upsert handler should return without error. Currently raises AttributeError
    on `request.id` in the return statement (server.py:37) — should be request.item.id."""
    from server import VectorStoreServicer

    with patch("server.VectorService") as MockService:
        MockService.return_value.add_item = MagicMock()
        servicer = VectorStoreServicer()

    request = vector_store_pb2.UpsertRequest(
        trace_id="repro-1",
        item=vector_store_pb2.UpsertItem(id=42, text="hello"),
    )

    response = servicer.Upsert(request, context=MagicMock())
    assert "42" in response.status
