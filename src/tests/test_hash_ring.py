"""
Tests for ConsistentHashRing, focused on get_nodes.

Run from project root:
    .venv/Scripts/python -m pytest src/tests/test_hash_ring.py -v
"""
from utils.hash_ring import ConsistentHashRing


def make_ring(*hosts):
    ring = ConsistentHashRing(virtual_nodes=50)
    for h in hosts:
        ring.add_node(h)
    return ring


def test_get_nodes_returns_n_distinct_hosts():
    ring = make_ring("a", "b", "c")
    result = get_nodes_for_any_key(ring, n=2)
    assert len(result) == 2
    assert len(set(result)) == 2  # no duplicates


def test_get_nodes_n_equals_total_returns_all_hosts():
    ring = make_ring("a", "b", "c")
    result = get_nodes_for_any_key(ring, n=3)
    assert set(result) == {"a", "b", "c"}


def test_get_nodes_n_exceeds_total_caps_at_available():
    ring = make_ring("a", "b")
    result = get_nodes_for_any_key(ring, n=10)
    assert set(result) == {"a", "b"}


def test_get_nodes_empty_ring_returns_empty():
    ring = ConsistentHashRing()
    assert ring.get_nodes("anything", 3) == []


def test_get_nodes_consistent_for_same_key():
    """Same key must always return the same ordered replica set."""
    ring = make_ring("a", "b", "c")
    first = ring.get_nodes("key-42", 2)
    second = ring.get_nodes("key-42", 2)
    assert first == second


def test_get_nodes_n1_matches_get_node():
    """get_nodes(key, 1) should return the same host as get_node(key)."""
    ring = make_ring("a", "b", "c")
    for key in ["0", "100", "999", "hello"]:
        assert ring.get_nodes(key, 1) == [ring.get_node(key)]


# Helper — picks a fixed key; we care about the count and uniqueness, not which key.
def get_nodes_for_any_key(ring, n):
    return ring.get_nodes("test-key", n)
