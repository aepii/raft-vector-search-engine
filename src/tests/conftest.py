"""
Shared fixtures and marker registration.

Two test tiers:
  - Unit tests (no marker) — pure Python, no network. Always runnable.
  - Integration tests (pytest.mark.integration) — require the docker compose
    cluster. Run: docker compose up -d, then pytest -m integration

Slow integration tests (pytest.mark.slow) take 30+ seconds (e.g. shard restart).
Run fast integration only: pytest -m "integration and not slow"
"""
import grpc
import pytest
import vector_store_pb2
import vector_store_pb2_grpc

COORDINATOR_ADDR = "localhost:50050"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: requires the docker compose cluster (run `docker compose up -d` first)",
    )
    config.addinivalue_line(
        "markers",
        "slow: integration tests that take 30+ seconds (shard restart / state transfer)",
    )


def _probe_coordinator() -> bool:
    try:
        ch = grpc.insecure_channel(
            COORDINATOR_ADDR, options=[("grpc.enable_retries", 0)]
        )
        stub = vector_store_pb2_grpc.VectorStoreStub(ch)
        stub.Count(vector_store_pb2.CountRequest(), timeout=2)
        ch.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def compose_cluster():
    """Skip the test if the docker compose cluster is not reachable."""
    if not _probe_coordinator():
        pytest.skip(
            "Docker compose cluster not reachable — run `docker compose up -d` first"
        )
