import os
import time
import threading
import grpc
from concurrent import futures
import vector_store_pb2, vector_store_pb2_grpc
from classes.vector_service import VectorService
from classes.vector_store import VectorStore
from utils.logger import get_logger
from dotenv import load_dotenv

load_dotenv()

SERVER_PORT = os.getenv("SERVER_PORT", "50051")
DB_PATH = os.getenv("DB_PATH", f"./data/shard_{SERVER_PORT}.db")
COORDINATOR_HOST = os.getenv("COORDINATOR_HOST", "localhost:50050")
# Advertised address sent in heartbeats. Override in Docker/Chameleon where
# the container's hostname differs from localhost.
SHARD_HOST = os.getenv("SHARD_HOST", f"localhost:{SERVER_PORT}")
HEARTBEAT_INTERVAL_S = int(os.getenv("HEARTBEAT_INTERVAL_S", "5"))

logger = get_logger(f"SHARD:{SERVER_PORT}")


class VectorStoreServicer(vector_store_pb2_grpc.VectorStoreServicer):
    """gRPC servicer implementation for the VectorStore service."""

    def __init__(self):
        # dirname returns "" if DB_PATH is a bare filename (e.g. DB_PATH=shard.db),
        # in which case the empty string "" is passed to os.makedirs
        # calling os.makedirs("") raises an error
        # this check prevents the error
        # probably not necessary, but safe
        dir_name = os.path.dirname(DB_PATH)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        self.service = VectorService(VectorStore(DB_PATH))

    def Upsert(self, request, context):
        """
        Inserts or updates an item in the vector store.
        """
        trace_id = request.trace_id
        item = request.item

        logger.info(f"[{trace_id}] [Upsert] id={item.id}")
        self.service.add_item(item.id, item.text, list(item.embedding))
        logger.info(f"[{trace_id}] [Upsert] id={item.id} indexed")

        return vector_store_pb2.UpsertResponse(status=f"ID {request.item.id} indexed.")

    def UpsertBatch(self, request, context):
        """
        Inserts or updates a batch of items in the vector store.
        """
        trace_id = request.trace_id

        logger.info(f"[{trace_id}] [UpsertBatch] size={len(request.items)}")

        items = [(item.id, item.text, list(item.embedding)) for item in request.items]
        t = time.perf_counter()
        self.service.add_items_batch(items)
        elapsed = (time.perf_counter() - t) * 1000

        logger.info(f"[{trace_id}] [UpsertBatch] indexed {len(items)} items in {elapsed:.0f}ms")

        statuses = [f"ID {item.id} indexed." for item in request.items]
        return vector_store_pb2.UpsertBatchResponse(statuses=statuses)

    def Count(self, request, context):
        count = self.service.vector_store.count()
        return vector_store_pb2.CountResponse(count=count)

    def Search(self, request, context):
        """
        Performs a similarity search over the indexed vectors.
        """
        trace_id = request.trace_id

        logger.info(f"[{trace_id}] [Search] query='{request.query_text}'")

        results = self.service.search(list(request.query_vector), top_k=request.top_k)

        logger.info(f"[{trace_id}] [Search] returning {len(results)} results")

        return vector_store_pb2.SearchResponse(
            results=[
                vector_store_pb2.SearchResult(text=text, score=score)
                for text, score in results
            ]
        )


def _heartbeat_loop():
    channel = grpc.insecure_channel(COORDINATOR_HOST)
    stub = vector_store_pb2_grpc.CoordinatorControlStub(channel)
    request = vector_store_pb2.HeartbeatRequest(host=SHARD_HOST)
    while True:
        try:
            stub.Heartbeat(request)
        except grpc.RpcError as e:
            logger.warning(f"Heartbeat to {COORDINATOR_HOST} failed: {e.details()}")
        time.sleep(HEARTBEAT_INTERVAL_S)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    vector_store_pb2_grpc.add_VectorStoreServicer_to_server(
        VectorStoreServicer(), server
    )
    server.add_insecure_port(f"[::]:{SERVER_PORT}")
    logger.info(f"Shard started on port {SERVER_PORT}")
    server.start()

    t = threading.Thread(target=_heartbeat_loop, daemon=True)
    t.start()
    logger.info(f"Heartbeat thread started -> {COORDINATOR_HOST} as {SHARD_HOST}")

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
