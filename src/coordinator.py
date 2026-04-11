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

logger = get_logger("COORDINATOR")


class CoordinatorServicer(vector_store_pb2_grpc.VectorStoreServicer):
    def __init__(self, shard_hosts: list[str]):
        self._lock = threading.RLock()
        self._ring = ConsistentHashRing(virtual_nodes=150)
        self._stub_map: dict[str, vector_store_pb2_grpc.VectorStoreStub] = {}
        self._embedding_model = EmbeddingModel()

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

    def _route(self, item_id: int) -> tuple[str, vector_store_pb2_grpc.VectorStoreStub]:
        """Map an item ID to a (host, stub) pair using the consistent hash ring."""
        with self._lock:
            host = self._ring.get_node(str(item_id))
            if host is None:
                raise RuntimeError("No shard nodes in ring")
            return host, self._stub_map[host]

    def Upsert(self, request, context):
        trace_id = request.trace_id
        item = request.item
        host, stub = self._route(item.id)

        logger.info(f"[{trace_id}] [Upsert] id={item.id} -> {host}")

        embedding = self._embedding_model.encode(item.text).tolist()
        encoded_request = vector_store_pb2.UpsertRequest(
            item=vector_store_pb2.UpsertItem(id=item.id, text=item.text, embedding=embedding),
            trace_id=trace_id,
        )
        response = stub.Upsert(encoded_request)

        logger.info(f"[{trace_id}] [Upsert] id={item.id} complete: {response.status}")

        return vector_store_pb2.UpsertResponse(status=f"[{host}] {response.status}")

    def UpsertBatch(self, request, context):
        trace_id = request.trace_id

        logger.info(f"[{trace_id}] [UpsertBatch] received size={len(request.items)}")

        texts = [item.text for item in request.items]
        embeddings = self._embedding_model.encode(texts)

        shard_batches: dict[str, list] = {}
        for item, embedding in zip(request.items, embeddings):
            host, _ = self._route(item.id)
            logger.info(f"[{trace_id}] item id={item.id} -> {host}")
            encoded_item = vector_store_pb2.UpsertItem(
                id=item.id, text=item.text, embedding=embedding.tolist()
            )
            shard_batches.setdefault(host, []).append(encoded_item)

        responses = []
        for host, batch in shard_batches.items():
            with self._lock:
                stub = self._stub_map.get(host)
            if stub is None:
                logger.warning(f"[{trace_id}] skipping {host}: not in stub map")
                continue

            logger.info(f"[{trace_id}] sending {len(batch)} items to {host}")

            batch_request = vector_store_pb2.UpsertBatchRequest(items=batch, trace_id=trace_id)

            start = time.perf_counter()
            response = stub.UpsertBatch(batch_request)
            elapsed = (time.perf_counter() - start) * 1000

            logger.info(f"[{trace_id}] {host} batch complete in {elapsed:.2f}ms")

            responses.extend(response.statuses)

        logger.info(f"[{trace_id}] [UpsertBatch] done, {len(responses)} total statuses")

        return vector_store_pb2.UpsertBatchResponse(statuses=responses)

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
        top = all_results[: request.top_k]

        logger.info(f"[{trace_id}] Search merged -> returning top {len(top)} results")
        return vector_store_pb2.SearchResponse(results=top)


class CoordinatorControlServicer(vector_store_pb2_grpc.CoordinatorControlServicer):
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
    coordinator = CoordinatorServicer(SHARD_HOSTS)

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
