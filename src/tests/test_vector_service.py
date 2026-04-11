"""
Unit tests for VectorService with pre-computed embeddings.

VectorService no longer owns an EmbeddingModel — it accepts pre-computed float
vectors from the coordinator and stores/searches them directly.

Run from src/:
    pytest tests/test_vector_service.py -v
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from classes.vector_service import VectorService


def test_add_item_stores_and_retrieves():
    """add_item accepts a pre-computed embedding; search returns the stored text."""
    service = VectorService()
    embedding = np.random.rand(384).tolist()
    service.add_item(1, "hello world", embedding)
    assert service.vector_store.metadata[1] == "hello world"
    assert service.vector_store.store[1].shape == (384,)


def test_search_returns_closest_vector():
    """search ranks the most similar vector first."""
    service = VectorService()
    dim = 4
    # item 1: unit vector along first axis
    e1 = [1.0, 0.0, 0.0, 0.0]
    # item 2: unit vector along second axis (orthogonal to e1)
    e2 = [0.0, 1.0, 0.0, 0.0]
    service.add_item(1, "item one", e1)
    service.add_item(2, "item two", e2)

    # query is close to item 1
    query = [0.9, 0.1, 0.0, 0.0]
    results = service.search(query, top_k=2)

    assert len(results) == 2
    assert results[0][0] == "item one"
    assert results[0][1] > results[1][1]


def test_add_items_batch_stores_all():
    """add_items_batch stores all items correctly."""
    service = VectorService()
    items = [
        (10, "alpha", np.random.rand(384).tolist()),
        (20, "beta", np.random.rand(384).tolist()),
        (30, "gamma", np.random.rand(384).tolist()),
    ]
    service.add_items_batch(items)
    assert len(service.vector_store.store) == 3
    assert service.vector_store.metadata[10] == "alpha"
    assert service.vector_store.metadata[20] == "beta"
    assert service.vector_store.metadata[30] == "gamma"


def test_search_top_k_limits_results():
    """search respects the top_k parameter."""
    service = VectorService()
    for i in range(10):
        service.add_item(i, f"item {i}", np.random.rand(384).tolist())

    results = service.search(np.random.rand(384).tolist(), top_k=3)
    assert len(results) == 3
