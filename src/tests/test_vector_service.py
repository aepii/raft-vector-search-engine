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
from classes.vector_store import VectorStore


def make_service(dim=384):
    return VectorService(VectorStore(dim=dim))


def test_add_item_stores_and_retrieves():
    """add_item accepts a pre-computed embedding; search returns the stored text."""
    service = make_service()
    embedding = np.random.rand(384).tolist()
    service.add_item(1, "hello world", embedding)
    results = service.search(embedding, top_k=1)
    assert results[0][0] == "hello world"


def test_search_returns_closest_vector():
    """search ranks the most similar vector first."""
    service = make_service(dim=4)
    e1 = [1.0, 0.0, 0.0, 0.0]
    e2 = [0.0, 1.0, 0.0, 0.0]
    service.add_item(1, "item one", e1)
    service.add_item(2, "item two", e2)

    query = [0.9, 0.1, 0.0, 0.0]
    results = service.search(query, top_k=2)

    assert len(results) == 2
    assert results[0][0] == "item one"
    assert results[0][1] > results[1][1]


def test_add_items_batch_stores_all():
    """add_items_batch stores all items; each text is retrievable via search."""
    service = make_service(dim=4)
    # Use orthogonal unit vectors so each item is its own nearest neighbour.
    items = [
        (10, "alpha", [1.0, 0.0, 0.0, 0.0]),
        (20, "beta",  [0.0, 1.0, 0.0, 0.0]),
        (30, "gamma", [0.0, 0.0, 1.0, 0.0]),
    ]
    service.add_items_batch(items)
    assert service.vector_store.count() == 3
    assert service.search([1.0, 0.0, 0.0, 0.0], top_k=1)[0][0] == "alpha"
    assert service.search([0.0, 1.0, 0.0, 0.0], top_k=1)[0][0] == "beta"
    assert service.search([0.0, 0.0, 1.0, 0.0], top_k=1)[0][0] == "gamma"


def test_search_top_k_limits_results():
    """search respects the top_k parameter."""
    service = make_service()
    for i in range(10):
        service.add_item(i, f"item {i}", np.random.rand(384).tolist())

    results = service.search(np.random.rand(384).tolist(), top_k=3)
    assert len(results) == 3
