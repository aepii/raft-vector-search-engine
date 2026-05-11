"""
Unit tests for VectorStore (sqlite-vec persistence).

Tests use in-memory databases (db_path=":memory:") and small embedding
dimensions (dim=4) for speed and isolation — no disk I/O, no cleanup needed.

Run from src/:
    pytest tests/test_vector_store.py -v
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from classes.vector_store import VectorStore


def make_store():
    return VectorStore(dim=4)


def test_upsert_stores_text():
    store = make_store()
    store.upsert(1, "hello", [0.1, 0.2, 0.3, 0.4])
    assert store.count() == 1


def test_upsert_overwrites_on_duplicate_id():
    store = make_store()
    store.upsert(1, "first", [1.0, 0.0, 0.0, 0.0])
    store.upsert(1, "second", [0.0, 1.0, 0.0, 0.0])
    assert store.count() == 1
    results = store.search([0.0, 1.0, 0.0, 0.0], top_k=1)
    assert results[0][0] == "second"


def test_count_reflects_stored_items():
    store = make_store()
    assert store.count() == 0
    store.upsert(1, "a", [1.0, 0.0, 0.0, 0.0])
    store.upsert(2, "b", [0.0, 1.0, 0.0, 0.0])
    assert store.count() == 2


def test_search_returns_closest_first():
    store = make_store()
    store.upsert(1, "item one", [1.0, 0.0, 0.0, 0.0])
    store.upsert(2, "item two", [0.0, 1.0, 0.0, 0.0])
    # query close to item one
    results = store.search([0.9, 0.1, 0.0, 0.0], top_k=2)
    assert results[0][0] == "item one"
    assert results[0][1] > results[1][1]


def test_score_is_cosine_similarity_not_distance():
    """Identical vectors should score 1.0; sqlite-vec returns distance so VectorStore
    must convert: score = 1 - distance."""
    store = make_store()
    store.upsert(1, "self", [1.0, 0.0, 0.0, 0.0])
    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert abs(results[0][1] - 1.0) < 1e-5


def test_upsert_is_atomic_items_and_vec_stay_in_sync():
    """Both tables should always reflect the same set of IDs."""
    store = make_store()
    for i in range(5):
        store.upsert(i, f"item {i}", [float(i == j) for j in range(4)])
    rows = store.conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    vecs = store.conn.execute("SELECT COUNT(*) FROM vec_items").fetchone()[0]
    assert rows == vecs == 5
