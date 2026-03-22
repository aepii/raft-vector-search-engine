import pytest
import sys
import os
from client.vector_store_client import VectorStoreClient


@pytest.fixture
def client():
    c = VectorStoreClient()
    yield c
    c.close()


def test_upsert_returns_status(client):
    status = client.upsert(999, "Test document")
    assert "999" in status


def test_search_returns_results(client):
    client.upsert(1, "I love dogs.")
    client.upsert(2, "Dogs are loyal animals.")
    client.upsert(3, "Cats are cool animals.")
    results = client.search("A dog running", top_k=2)
    assert len(results) == 2


def test_semantic_similarity(client):
    client.upsert(10, "Python is a great programming language!")
    client.upsert(11, "Java is a great programming language!")
    client.upsert(9, "Python snake!")
    results = client.search("Coding in python", top_k=1)
    assert results[0] == "Python is a great programming language!"
