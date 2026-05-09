"""
State transfer tests — two tiers.

────────────────────────────────────────────────────────────────
TIER 1 — Unit tests (no running processes required)
────────────────────────────────────────────────────────────────
    pytest tests/test_state_transfer.py -v -k "unit"

────────────────────────────────────────────────────────────────
TIER 2 — Integration tests (docker compose cluster required)
────────────────────────────────────────────────────────────────
    docker compose up -d
    pytest tests/test_state_transfer.py -v -m integration

TIER 2 includes a shard-restart test (pytest.mark.slow) that takes ~35s:
it stops shard-2, waits for the coordinator to deregister it, restarts it,
and asserts that state transfer restores the data. Requires docker and
docker compose on the PATH.

    pytest tests/test_state_transfer.py -v -m "integration and slow"

Run from src/:
    pytest tests/test_state_transfer.py -v
"""
import hashlib
import subprocess
import sys
import os
import time
from pathlib import Path

import grpc
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from classes.vector_store import VectorStore
from coordinator import CoordinatorServicer, CoordinatorControlServicer
import vector_store_pb2
import vector_store_pb2_grpc

COORDINATOR = "localhost:50050"
SHARD_51 = "localhost:50051"
SHARD_53 = "localhost:50053"

# docker-compose.yml lives two levels up from src/tests/
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Small fixed dataset used across tiers — synthetic embeddings, dim=4 for unit tests.
UNIT_ITEMS = [
    (1, "apple",  [1.0, 0.0, 0.0, 0.0]),
    (2, "banana", [0.0, 1.0, 0.0, 0.0]),
    (3, "cherry", [0.0, 0.0, 1.0, 0.0]),
    (4, "date",   [0.0, 0.0, 0.0, 1.0]),
]

# Items used in Tier 2 via the live coordinator.
INTEGRATION_ITEMS = [
    (201, "state transfer test item one"),
    (202, "state transfer test item two"),
    (203, "state transfer test item three"),
    (204, "state transfer test item four"),
    (205, "state transfer test item five"),
]
INTEGRATION_COUNT = len(INTEGRATION_ITEMS)


# ─── Tier 1: unit tests ───────────────────────────────────────────────────────

class TestScanUnit:
    """VectorStore.scan() with an in-memory DB — no network required."""

    def test_scan_empty_store(self):
        store = VectorStore(dim=4)
        assert store.scan() == []

    def test_scan_returns_all_items(self):
        store = VectorStore(dim=4)
        for item_id, text, emb in UNIT_ITEMS:
            store.upsert(item_id, text, emb)
        rows = store.scan()
        assert len(rows) == len(UNIT_ITEMS)
        ids = {r[0] for r in rows}
        assert ids == {item[0] for item in UNIT_ITEMS}

    def test_scan_embeddings_round_trip(self):
        store = VectorStore(dim=4)
        store.upsert(1, "hello", [0.25, 0.5, 0.75, 1.0])
        rows = store.scan()
        assert len(rows) == 1
        item_id, text, emb = rows[0]
        assert item_id == 1
        assert text == "hello"
        assert all(abs(a - b) < 1e-5 for a, b in zip(emb, [0.25, 0.5, 0.75, 1.0]))

    def test_scan_consistent_with_count(self):
        store = VectorStore(dim=4)
        for item_id, text, emb in UNIT_ITEMS:
            store.upsert(item_id, text, emb)
        assert len(store.scan()) == store.count()


class TestGetPeersArcMathUnit:
    """GetPeers arc assignment logic — no network required."""

    def _make_servicer(self, hosts: list[str]) -> CoordinatorControlServicer:
        coord = CoordinatorServicer(
            shard_hosts=hosts,
            replication_factor=0,
            heartbeat_interval=99999,
            heartbeat_timeout=99999,
        )
        return CoordinatorControlServicer(coord)

    def test_no_peers_returns_empty(self):
        svc = self._make_servicer([])
        resp = svc.GetPeers(
            vector_store_pb2.GetPeersRequest(host="localhost:50051"), context=None
        )
        assert resp.peers == []

    def test_single_peer_covers_full_space(self):
        svc = self._make_servicer(["localhost:50052"])
        resp = svc.GetPeers(
            vector_store_pb2.GetPeersRequest(host="localhost:50051"), context=None
        )
        assert len(resp.peers) == 1
        peer = resp.peers[0]
        assert peer.start_hash == "0" * 64
        assert peer.end_hash == ""  # no upper bound = end of space

    def test_caller_is_excluded(self):
        svc = self._make_servicer(["localhost:50051", "localhost:50052"])
        resp = svc.GetPeers(
            vector_store_pb2.GetPeersRequest(host="localhost:50051"), context=None
        )
        hosts = [p.host for p in resp.peers]
        assert "localhost:50051" not in hosts
        assert "localhost:50052" in hosts

    def test_arcs_are_contiguous_and_cover_full_space(self):
        svc = self._make_servicer(["localhost:50051", "localhost:50052", "localhost:50053"])
        resp = svc.GetPeers(
            vector_store_pb2.GetPeersRequest(host="localhost:50099"), context=None
        )
        peers = resp.peers
        assert len(peers) == 3

        # First arc starts at zero.
        assert peers[0].start_hash == "0" * 64

        # Each arc's end equals the next arc's start.
        for i in range(len(peers) - 1):
            assert peers[i].end_hash == peers[i + 1].start_hash, (
                f"Gap between arc {i} and {i+1}"
            )

        # Last arc has no upper bound.
        assert peers[-1].end_hash == ""

    def test_arc_hashes_are_64_char_hex(self):
        svc = self._make_servicer(["localhost:50052", "localhost:50053"])
        resp = svc.GetPeers(
            vector_store_pb2.GetPeersRequest(host="localhost:50051"), context=None
        )
        for peer in resp.peers:
            if peer.end_hash:
                assert len(peer.end_hash) == 64
                int(peer.end_hash, 16)  # raises if not valid hex
            assert len(peer.start_hash) == 64
            int(peer.start_hash, 16)


# ─── Tier 2: integration tests ────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ctrl_stub(compose_cluster):
    return vector_store_pb2_grpc.CoordinatorControlStub(
        grpc.insecure_channel(COORDINATOR)
    )


@pytest.fixture(scope="module")
def coord_stub(compose_cluster):
    return vector_store_pb2_grpc.VectorStoreStub(grpc.insecure_channel(COORDINATOR))


@pytest.fixture(scope="module")
def shard51_stub(compose_cluster):
    return vector_store_pb2_grpc.VectorStoreStub(grpc.insecure_channel(SHARD_51))


@pytest.mark.integration
class TestGetPeersIntegration:
    """GetPeers over the wire against a live coordinator."""

    def test_get_peers_excludes_caller(self, ctrl_stub):
        resp = ctrl_stub.GetPeers(
            vector_store_pb2.GetPeersRequest(host=SHARD_51)
        )
        hosts = [p.host for p in resp.peers]
        assert SHARD_51 not in hosts

    def test_get_peers_returns_registered_shards(self, ctrl_stub):
        resp = ctrl_stub.GetPeers(
            vector_store_pb2.GetPeersRequest(host="localhost:59999")
        )
        assert len(resp.peers) >= 2

    def test_get_peers_arcs_contiguous(self, ctrl_stub):
        resp = ctrl_stub.GetPeers(
            vector_store_pb2.GetPeersRequest(host="localhost:59999")
        )
        peers = resp.peers
        if len(peers) < 2:
            pytest.skip("Need at least 2 donors to check contiguity")
        for i in range(len(peers) - 1):
            assert peers[i].end_hash == peers[i + 1].start_hash


@pytest.mark.integration
class TestDumpIntegration:
    """Dump RPC against a live shard — seeds a few items, dumps them back."""

    @pytest.fixture(scope="class", autouse=True)
    def seed_items(self, coord_stub):
        """Upsert a small fixed set through the coordinator before dump tests."""
        items = [
            vector_store_pb2.UpsertItem(id=item_id, text=text)
            for item_id, text in INTEGRATION_ITEMS
        ]
        coord_stub.UpsertBatch(
            vector_store_pb2.UpsertBatchRequest(items=items, trace_id="test-dump-seed")
        )
        time.sleep(0.2)  # let writes land

    def test_dump_full_space_returns_items(self, shard51_stub):
        items = list(shard51_stub.Dump(
            vector_store_pb2.DumpRequest(start_hash="0" * 64, end_hash="")
        ))
        assert len(items) > 0

    def test_dump_items_have_id_text_embedding(self, shard51_stub):
        items = list(shard51_stub.Dump(
            vector_store_pb2.DumpRequest(start_hash="0" * 64, end_hash="")
        ))
        for item in items:
            assert item.id > 0
            assert item.text
            assert len(item.embedding) == 384

    def test_dump_arc_is_subset_of_full(self, shard51_stub):
        """Items in the first half of the hash space should be a subset of all items."""
        midpoint = "7" + "f" * 63
        all_ids = {
            item.id for item in shard51_stub.Dump(
                vector_store_pb2.DumpRequest(start_hash="0" * 64, end_hash="")
            )
        }
        first_half_ids = {
            item.id for item in shard51_stub.Dump(
                vector_store_pb2.DumpRequest(start_hash="0" * 64, end_hash=midpoint)
            )
        }
        assert first_half_ids.issubset(all_ids)

    def test_dump_arc_bounds_are_exclusive_upper(self, shard51_stub):
        """Items in [0, mid) and [mid, end) together equal all items."""
        midpoint = format(2 ** 255, "064x")
        lower = {
            item.id for item in shard51_stub.Dump(
                vector_store_pb2.DumpRequest(start_hash="0" * 64, end_hash=midpoint)
            )
        }
        upper = {
            item.id for item in shard51_stub.Dump(
                vector_store_pb2.DumpRequest(start_hash=midpoint, end_hash="")
            )
        }
        all_ids = {
            item.id for item in shard51_stub.Dump(
                vector_store_pb2.DumpRequest(start_hash="0" * 64, end_hash="")
            )
        }
        assert lower | upper == all_ids
        assert lower & upper == set()  # no overlap


@pytest.mark.integration
@pytest.mark.slow
class TestStateTransferRestart:
    """
    Full state-transfer cycle: seed data, stop a shard, restart it, verify recovery.

    Orchestrates the docker compose cluster via subprocess. Requires docker and
    docker compose on the PATH. Takes ~35s (15s for coordinator to deregister the
    shard + transfer time).
    """

    def test_shard_recovers_data_after_restart(self, coord_stub):
        # Seed data through the coordinator.
        items = [
            vector_store_pb2.UpsertItem(id=item_id, text=text)
            for item_id, text in INTEGRATION_ITEMS
        ]
        coord_stub.UpsertBatch(
            vector_store_pb2.UpsertBatchRequest(items=items, trace_id="restart-seed")
        )
        time.sleep(1)

        # Stop shard-2 (port 50053 in docker-compose.yml).
        subprocess.run(
            ["docker", "compose", "stop", "shard-2"],
            cwd=str(PROJECT_ROOT),
            check=True,
        )

        # Wait for coordinator to deregister (default HEARTBEAT_TIMEOUT_S = 15s).
        time.sleep(20)

        # Restart — state transfer runs automatically before re-registration.
        subprocess.run(
            ["docker", "compose", "start", "shard-2"],
            cwd=str(PROJECT_ROOT),
            check=True,
        )

        # Poll until the shard reports the expected item count (up to 30s).
        shard = vector_store_pb2_grpc.VectorStoreStub(grpc.insecure_channel(SHARD_53))
        deadline = time.time() + 30
        recovered = False
        while time.time() < deadline:
            try:
                count = shard.Count(vector_store_pb2.CountRequest(), timeout=2).count
                if count >= INTEGRATION_COUNT:
                    recovered = True
                    break
            except grpc.RpcError:
                pass
            time.sleep(1)

        assert recovered, (
            f"Shard did not recover {INTEGRATION_COUNT} items within 30s after restart"
        )

    def test_recovered_shard_search_returns_results(self):
        """Transferred data should be searchable, not just counted."""
        shard = vector_store_pb2_grpc.VectorStoreStub(grpc.insecure_channel(SHARD_53))
        resp = shard.Search(
            vector_store_pb2.SearchRequest(
                query_text="state transfer test item",
                query_vector=[0.1] * 384,
                top_k=3,
                trace_id="restart-search",
            )
        )
        assert len(resp.results) > 0, (
            "Shard returned no search results after state transfer"
        )
