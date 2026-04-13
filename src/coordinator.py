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

SHARD_HOSTS = os.getenv(
    "SHARD_HOSTS", "localhost:50051,localhost:50052,localhost:50053"
).split(",")
COORDINATOR_PORT = os.getenv("COORDINATOR_PORT", "50050")
# Number of shards that store each item. 0 means all registered nodes (full replication).
REPLICATION_FACTOR = int(os.getenv("REPLICATION_FACTOR", "0"))

logger = get_logger("COORDINATOR")


class CoordinatorServicer(vector_store_pb2_grpc.VectorStoreServicer):
    """
    gRPC servicer for the VectorStore service, running on the coordinator.

    Encodes all text and queries via a single EmbeddingModel instance before
    forwarding pre-computed vectors to shards. Writes are routed to a single
    shard via a consistent hash ring; searches fan out to all registered shards
    in parallel and the results are merged and re-ranked before returning.
    """

    def __init__(self, shard_hosts: list[str], replication_factor: int = 0):
        self._lock = threading.RLock()
        self._ring = ConsistentHashRing(virtual_nodes=150)
        self._stub_map: dict[str, vector_store_pb2_grpc.VectorStoreStub] = {}
        self._embedding_model = EmbeddingModel()
        # 0 means replicate to all registered nodes.
        self._replication_factor = replication_factor

        for host in (h.strip() for h in shard_hosts if h.strip()):
            self._add_node_locked(host)

        logger.info(f"Coordinator ready - {len(self._stub_map)} shards: {list(self._stub_map)}")

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

    Handles runtime node registration and deregistration by mutating the
    CoordinatorServicer's hash ring and stub map.
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

        # NOTE: no data migration. Historical writes on the deregistered node
        # stay there; Search still reaches them if the process is running.
        # New writes for keys that mapped to this node now go to the next
        # clockwise node on the ring.
        logger.info(f"[DeregisterNode] {host} removed (ring size={count})")
        return vector_store_pb2.NodeResponse(
            success=True, message=f"{host} removed", node_count=count
        )


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
