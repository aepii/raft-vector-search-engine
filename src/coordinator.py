import os
import time
import threading
import grpc
from concurrent import futures
import vector_store_pb2, vector_store_pb2_grpc
from utils.hash_ring import ConsistentHashRing
from utils.logger import get_logger
from classes.embedding_model import EmbeddingModel
from dotenv import load_dotenv

load_dotenv()

SHARD_HOSTS = os.getenv("SHARD_HOSTS", "").split(",")
COORDINATOR_PORT = os.getenv("COORDINATOR_PORT", "50050")
# Number of shards that store each item. 0 means all registered nodes (full replication).
REPLICATION_FACTOR = int(os.getenv("REPLICATION_FACTOR", "0"))
HEARTBEAT_INTERVAL_S = int(os.getenv("HEARTBEAT_INTERVAL_S", "5"))
HEARTBEAT_TIMEOUT_S = int(os.getenv("HEARTBEAT_TIMEOUT_S", "15"))

logger = get_logger("COORDINATOR")


class CoordinatorServicer(vector_store_pb2_grpc.VectorStoreServicer):
    """
    gRPC servicer for the VectorStore service, running on the coordinator.

    Encodes all text and queries via a single EmbeddingModel instance before
    forwarding pre-computed vectors to shards. Writes fan out to N clockwise
    ring successors (REPLICATION_FACTOR env var; default 0 = all nodes). Searches
    fan out to all registered shards in parallel; results are merged, deduplicated
    by text, and re-ranked before returning.
    """

    def __init__(
        self,
        shard_hosts: list[str],
        replication_factor: int = 0,
        heartbeat_interval: int = HEARTBEAT_INTERVAL_S,
        heartbeat_timeout: int = HEARTBEAT_TIMEOUT_S,
    ):
        self._lock = threading.RLock()
        self._ring = ConsistentHashRing(virtual_nodes=150)
        self._stub_map: dict[str, vector_store_pb2_grpc.VectorStoreStub] = {}
        self._last_seen: dict[str, float] = {}  # host -> unix timestamp of last heartbeat
        self._embedding_model = EmbeddingModel()
        self._replication_factor = replication_factor
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout

        for host in (h.strip() for h in shard_hosts if h.strip()):
            self._add_node_locked(host)

        sweep = threading.Thread(target=self._sweep_loop, daemon=True)
        sweep.start()

        logger.info(f"Coordinator ready - {len(self._stub_map)} shards: {list(self._stub_map)}")

    def _sweep_loop(self) -> None:
        """Periodically deregister shards that have stopped sending heartbeats."""
        while True:
            time.sleep(self._heartbeat_interval)
            now = time.time()
            with self._lock:
                stale = [
                    host for host, ts in self._last_seen.items()
                    if now - ts > self._heartbeat_timeout and host in self._stub_map
                ]
                for host in stale:
                    self._ring.remove_node(host)
                    del self._stub_map[host]
                    logger.warning(f"[Health] {host} timed out, deregistered (ring size={len(self._stub_map)})")

    def _add_node_locked(self, host: str) -> None:
        """Must be called with self._lock held (or during __init__)."""
        if host in self._stub_map:
            return
        self._stub_map[host] = vector_store_pb2_grpc.VectorStoreStub(grpc.insecure_channel(host))
        self._ring.add_node(host)
        logger.info(f"Node added: {host} (ring size={len(self._ring)})")

    def _replication_targets(self, key: str) -> list[tuple[str, vector_store_pb2_grpc.VectorStoreStub]]:
        """Return the (host, stub) pairs that should store the given key.

        Uses the ring to find the N clockwise successors from the key's hash
        position, where N is self._replication_factor (0 = all nodes).
        """
        with self._lock:
            n = self._replication_factor or len(self._ring)
            hosts = self._ring.get_nodes(key, n)
            return [(host, self._stub_map[host]) for host in hosts]

    def Upsert(self, request, context):
        trace_id = request.trace_id
        item = request.item

        embedding = self._embedding_model.encode(item.text).tolist()
        encoded_request = vector_store_pb2.UpsertRequest(
            item=vector_store_pb2.UpsertItem(id=item.id, text=item.text, embedding=embedding),
            trace_id=trace_id,
        )

        targets = self._replication_targets(str(item.id))

        logger.info(f"[{trace_id}] [Upsert] id={item.id} -> {len(targets)} shards")

        # Fan out synchronously (parallel via ThreadPoolExecutor, not fire-and-forget).
        # Async replication would let a slow or failed shard go undetected, leaving it
        # with stale data that it would serve on future searches. Sync fan-out means
        # write latency is max(shard latencies), not sum — same model as Search.
        def write_shard(host_and_stub):
            host, stub = host_and_stub
            try:
                response = stub.Upsert(encoded_request)
                logger.info(f"[{trace_id}] [Upsert] id={item.id} -> {host}: {response.status}")
                return response.status
            except grpc.RpcError as e:
                logger.error(f"[{trace_id}] [Upsert] id={item.id} -> {host} failed: {e.details()}")
                return None

        statuses = []
        with futures.ThreadPoolExecutor(max_workers=max(len(targets), 1)) as executor:
            for status in executor.map(write_shard, targets):
                if status is not None:
                    statuses.append(status)

        logger.info(f"[{trace_id}] [Upsert] id={item.id} complete: {len(statuses)}/{len(targets)} shards confirmed")
        return vector_store_pb2.UpsertResponse(status=f"replicated to {len(statuses)}/{len(targets)} shards")

    def UpsertBatch(self, request, context):
        trace_id = request.trace_id

        logger.info(f"[{trace_id}] [UpsertBatch] received size={len(request.items)}")

        texts = [item.text for item in request.items]
        t_enc = time.perf_counter()
        embeddings = self._embedding_model.encode(texts)
        enc_ms = (time.perf_counter() - t_enc) * 1000

        # Route each item to its replica set and group into per-shard batches.
        # With full replication every item lands on every shard; with N < total,
        # different items may land on different subsets.
        shard_batches: dict[str, list] = {}
        shard_stubs: dict[str, vector_store_pb2_grpc.VectorStoreStub] = {}
        for item, embedding in zip(request.items, embeddings):
            encoded_item = vector_store_pb2.UpsertItem(
                id=item.id, text=item.text, embedding=embedding.tolist()
            )
            for host, stub in self._replication_targets(str(item.id)):
                shard_batches.setdefault(host, []).append(encoded_item)
                shard_stubs[host] = stub

        logger.info(f"[{trace_id}] [UpsertBatch] encoded in {enc_ms:.0f}ms, routing to {len(shard_batches)} shards")

        def write_shard(host):
            stub = shard_stubs[host]
            batch = shard_batches[host]
            try:
                start = time.perf_counter()
                response = stub.UpsertBatch(
                    vector_store_pb2.UpsertBatchRequest(items=batch, trace_id=trace_id)
                )
                elapsed = (time.perf_counter() - start) * 1000
                logger.info(f"[{trace_id}] {host} batch ({len(batch)} items) complete in {elapsed:.2f}ms")
                return list(response.statuses)
            except grpc.RpcError as e:
                logger.error(f"[{trace_id}] {host} batch failed: {e.details()}")
                return []

        all_statuses = []
        with futures.ThreadPoolExecutor(max_workers=max(len(shard_batches), 1)) as executor:
            for statuses in executor.map(write_shard, shard_batches):
                all_statuses.extend(statuses)

        logger.info(f"[{trace_id}] [UpsertBatch] done: {len(request.items)} items x {len(shard_batches)} shards ({len(all_statuses)} acks)")
        return vector_store_pb2.UpsertBatchResponse(statuses=all_statuses)

    def Search(self, request, context):
        trace_id = request.trace_id

        logger.info(
            f"[{trace_id}] [Search] query='{request.query_text[:40]}' top_k={request.top_k}"
        )

        query_vector = self._embedding_model.encode(request.query_text).tolist()
        encoded_request = vector_store_pb2.SearchRequest(
            query_text=request.query_text,
            top_k=request.top_k,
            trace_id=trace_id,
            query_vector=query_vector,
        )

        with self._lock:
            stub_snapshot = list(self._stub_map.items())

        # Always fan out to every shard, even under full replication where any one
        # shard holds the complete dataset. Routing to a single shard would be more
        # efficient, but if that shard is down the query fails even though all others
        # are healthy. Fan-out + dedup preserves read fault tolerance at the cost of
        # redundant work, which is the right tradeoff here.
        def query_shard(host_and_stub):
            host, stub = host_and_stub
            try:
                start = time.perf_counter()
                response = stub.Search(encoded_request)
                elapsed = (time.perf_counter() - start) * 1000

                logger.info(
                    f"[{trace_id}] {host} returned {len(response.results)} results in {elapsed:.2f}ms"
                )
                return list(response.results)
            except grpc.RpcError as e:
                logger.error(f"[{trace_id}] {host} unavailable: {e.details()}")
                return []

        all_results = []
        with futures.ThreadPoolExecutor(max_workers=max(len(stub_snapshot), 1)) as executor:
            for result_list in executor.map(query_shard, stub_snapshot):
                all_results.extend(result_list)

        all_results.sort(key=lambda x: x.score, reverse=True)

        # Deduplicate by text. With full replication every shard holds every item,
        # so the same (text, score) appears once per shard in the merged list.
        # After sorting descending the first occurrence of each text is the
        # highest-scoring one; subsequent duplicates are dropped.
        seen_texts: set[str] = set()
        deduped = []
        for r in all_results:
            if r.text not in seen_texts:
                seen_texts.add(r.text)
                deduped.append(r)

        top = deduped[: request.top_k]

        logger.info(f"[{trace_id}] Search merged -> returning top {len(top)} results")
        return vector_store_pb2.SearchResponse(results=top)


class CoordinatorControlServicer(vector_store_pb2_grpc.CoordinatorControlServicer):
    """
    gRPC servicer for the CoordinatorControl service.

    Handles heartbeats, runtime node registration, and deregistration by mutating
    the CoordinatorServicer's hash ring and stub map. Under normal operation shards
    register implicitly via their first Heartbeat — RegisterNode is retained for
    manual bootstrapping only.
    """

    def __init__(self, coordinator: CoordinatorServicer):
        self._c = coordinator

    def RegisterNode(self, request, context):
        host = request.host.strip()
        if not host:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("host must not be empty")
            return vector_store_pb2.NodeResponse(success=False, message="empty host", node_count=0)

        with self._c._lock:
            already = host in self._c._stub_map
            if not already:
                self._c._add_node_locked(host)
            count = len(self._c._stub_map)

        msg = f"{'already registered' if already else 'registered'}: {host}"
        logger.info(f"[RegisterNode] {msg} (total nodes: {count})")
        return vector_store_pb2.NodeResponse(success=True, message=msg, node_count=count)

    def Heartbeat(self, request, context):
        host = request.host.strip()
        if not host:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("host must not be empty")
            return vector_store_pb2.HeartbeatResponse(registered=False)

        with self._c._lock:
            self._c._last_seen[host] = time.time()
            was_registered = host in self._c._stub_map
            if not was_registered:
                # Handles both initial registration and re-registration after
                # the sweep has deregistered a shard that was temporarily unreachable.
                self._c._add_node_locked(host)
                logger.info(f"[Heartbeat] {host} registered via heartbeat (total nodes: {len(self._c._stub_map)})")

        return vector_store_pb2.HeartbeatResponse(registered=True)

    def DeregisterNode(self, request, context):
        host = request.host.strip()

        with self._c._lock:
            if host not in self._c._stub_map:
                return vector_store_pb2.NodeResponse(
                    success=False,
                    message=f"{host} not found",
                    node_count=len(self._c._stub_map),
                )
            self._c._ring.remove_node(host)
            del self._c._stub_map[host]
            count = len(self._c._stub_map)

        # NOTE: no data migration. Data on the deregistered node's disk stays
        # there, but the node is removed from stub_map so Search will no longer
        # query it. Under full replication this is fine — all remaining shards
        # hold the same data. New writes route to the next clockwise node.
        logger.info(f"[DeregisterNode] {host} removed (ring size={count})")
        return vector_store_pb2.NodeResponse(
            success=True, message=f"{host} removed", node_count=count
        )

    def GetPeers(self, request, context):
        """Returns registered shards with evenly divided ring arc assignments.

        The calling shard passes its own host so it can be excluded. The SHA-256
        hash space [0, 2^256) is split into equal arcs, one per donor, for
        parallel state transfer. Under full replication every donor holds every
        key, so arc assignment is load distribution only — any healthy shard can
        provide any arc.
        """
        exclude = request.host.strip()
        with self._c._lock:
            donors = [h for h in self._c._stub_map if h != exclude]

        if not donors:
            return vector_store_pb2.GetPeersResponse(peers=[])

        space = 2 ** 256
        n = len(donors)
        arc_size = space // n
        assignments = []
        for i, host in enumerate(donors):
            start = format(i * arc_size, "064x")
            end = format((i + 1) * arc_size, "064x") if i < n - 1 else ""
            assignments.append(
                vector_store_pb2.PeerAssignment(host=host, start_hash=start, end_hash=end)
            )

        logger.info(f"[GetPeers] returning {len(assignments)} donors to {exclude or '(unspecified)'}")
        return vector_store_pb2.GetPeersResponse(peers=assignments)


def serve():
    coordinator = CoordinatorServicer(SHARD_HOSTS, replication_factor=REPLICATION_FACTOR)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    vector_store_pb2_grpc.add_VectorStoreServicer_to_server(coordinator, server)
    vector_store_pb2_grpc.add_CoordinatorControlServicer_to_server(
        CoordinatorControlServicer(coordinator), server
    )

    server.add_insecure_port(f"[::]:{COORDINATOR_PORT}")
    logger.info(f"Coordinator started on port {COORDINATOR_PORT}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
