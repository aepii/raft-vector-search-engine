"""
Unit tests verifying that the coordinator encodes text before forwarding to shards.

These tests mock EmbeddingModel and the shard stubs so no live processes are needed.
They verify that:
  - EmbeddingModel.encode is called with the correct text
  - The forwarded UpsertRequest / SearchRequest carries the pre-computed embedding

Run from src/:
    pytest tests/test_coordinator_encoding.py -v
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import vector_store_pb2


FAKE_EMBEDDING = np.array([0.1, 0.2, 0.3, 0.4])
FAKE_EMBEDDING_LIST = FAKE_EMBEDDING.tolist()


def _make_coordinator():
    """Return a CoordinatorServicer with mocked EmbeddingModel and no real shards."""
    with patch("coordinator.EmbeddingModel") as MockEmbeddingModel:
        mock_model = MagicMock()
        mock_model.encode.return_value = FAKE_EMBEDDING
        MockEmbeddingModel.return_value = mock_model

        from coordinator import CoordinatorServicer
        coord = CoordinatorServicer(shard_hosts=[])

    return coord, mock_model


def _add_fake_shard(coord):
    """Inject a fake shard stub into the coordinator's ring and stub map."""
    fake_stub = MagicMock()
    fake_stub.Upsert.return_value = vector_store_pb2.UpsertResponse(status="ID 1 indexed.")
    fake_stub.UpsertBatch.return_value = vector_store_pb2.UpsertBatchResponse(statuses=["ID 1 indexed."])
    fake_stub.Search.return_value = vector_store_pb2.SearchResponse(results=[])

    host = "localhost:50099"
    coord._ring.add_node(host)
    coord._stub_map[host] = fake_stub
    return fake_stub


def test_upsert_calls_encode_and_forwards_embedding():
    """Coordinator encodes text and the forwarded UpsertItem has embedding populated."""
    coord, mock_model = _make_coordinator()
    fake_stub = _add_fake_shard(coord)

    request = vector_store_pb2.UpsertRequest(
        trace_id="t1",
        item=vector_store_pb2.UpsertItem(id=1, text="hello"),
    )
    coord.Upsert(request, context=MagicMock())

    mock_model.encode.assert_called_once_with("hello")

    forwarded = fake_stub.Upsert.call_args[0][0]
    assert list(forwarded.item.embedding) == pytest.approx(FAKE_EMBEDDING_LIST)


def test_upsert_batch_calls_encode_once_for_all_texts():
    """Coordinator batch-encodes all texts in one call."""
    batch_embeddings = np.array([FAKE_EMBEDDING, FAKE_EMBEDDING])

    with patch("coordinator.EmbeddingModel") as MockEmbeddingModel:
        mock_model = MagicMock()
        mock_model.encode.return_value = batch_embeddings
        MockEmbeddingModel.return_value = mock_model

        from coordinator import CoordinatorServicer
        coord = CoordinatorServicer(shard_hosts=[])

    fake_stub = _add_fake_shard(coord)

    request = vector_store_pb2.UpsertBatchRequest(
        trace_id="t2",
        items=[
            vector_store_pb2.UpsertItem(id=1, text="alpha"),
            vector_store_pb2.UpsertItem(id=2, text="beta"),
        ],
    )
    coord.UpsertBatch(request, context=MagicMock())

    mock_model.encode.assert_called_once_with(["alpha", "beta"])


def test_search_calls_encode_and_forwards_query_vector():
    """Coordinator encodes query_text and the forwarded SearchRequest has query_vector."""
    coord, mock_model = _make_coordinator()
    fake_stub = _add_fake_shard(coord)

    request = vector_store_pb2.SearchRequest(
        trace_id="t3",
        query_text="what is python",
        top_k=3,
    )
    coord.Search(request, context=MagicMock())

    mock_model.encode.assert_called_once_with("what is python")

    forwarded = fake_stub.Search.call_args[0][0]
    assert list(forwarded.query_vector) == pytest.approx(FAKE_EMBEDDING_LIST)
    assert forwarded.query_text == "what is python"
