import sys
import os
import numpy as np
from classes.vector_store import VectorStore

def test_upsert_stores_embedding():
    store = VectorStore()
    embedding = np.array([0.1, 0.2, 0.3])
    store.upsert(1, "hello", embedding)
    assert 1 in store.store
    assert store.metadata[1] == "hello"

def test_upsert_overwrites_existing():
    store = VectorStore()
    store.upsert(1, "first", np.array([0.1]))
    store.upsert(1, "second", np.array([0.9]))
    assert store.metadata[1] == "second"