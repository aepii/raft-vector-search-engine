import hashlib
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
SKIP_STATE_TRANSFER = os.getenv("SKIP_STATE_TRANSFER", "").lower() in ("1", "true", "yes")

logger = get_logger(f"SHARD:{SERVER_PORT}")


class VectorStoreServicer(vector_store_pb2_grpc.VectorStoreServicer):
    """gRPC servicer for the VectorStore service, running on shards only.

    Stores pre-computed embeddings received from the coordinator. Also implements
    Dump for state transfer: a recovering shard can stream a slice of this shard's
    dataset to seed its own DB before joining the routing table.
    """

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

    def Dump(self, request, context):
        """Streams all locally stored items whose key hash falls in the requested arc.

        start_hash is inclusive; end_hash is exclusive. An empty end_hash means no
        upper bound (the final arc of the ring). Arc bounds are SHA-256 hex strings
        of str(item_id), which are always 64 chars and thus lexicographically ordered.

        The key hash matches the ring's write routing key (SHA-256 of str(item_id)),
        so arc boundaries aren't arbitrary — under partial replication they correspond
        to actual ownership ranges on the ring.
        """
        start_hash = request.start_hash
        end_hash = request.end_hash
        count = 0
        for item_id, text, embedding in self.service.vector_store.scan():
            key_hash = hashlib.sha256(str(item_id).encode()).hexdigest()
            in_arc = (
                key_hash >= start_hash
                if not end_hash
                else start_hash <= key_hash < end_hash
            )
            if in_arc:
                count += 1
                yield vector_store_pb2.DumpItem(id=item_id, text=text, embedding=embedding)
        logger.info(f"[Dump] streamed {count} items for arc [{start_hash[:8]}..., {end_hash[:8] if end_hash else 'end'})")


def _heartbeat_loop():
    # Sends a heartbeat to the coordinator every HEARTBEAT_INTERVAL_S seconds.
    # The first successful beat registers this shard; subsequent beats keep it
    # in the coordinator's routing table. Failures warn but do not crash — the
    # coordinator may not be up yet, or may restart; the loop retries automatically.
    channel = grpc.insecure_channel(COORDINATOR_HOST)
    stub = vector_store_pb2_grpc.CoordinatorControlStub(channel)
    request = vector_store_pb2.HeartbeatRequest(host=SHARD_HOST)
    while True:
        try:
            stub.Heartbeat(request)
        except grpc.RpcError as e:
            logger.warning(f"Heartbeat to {COORDINATOR_HOST} failed: {e.details()}")
        time.sleep(HEARTBEAT_INTERVAL_S)


def _state_transfer(servicer: VectorStoreServicer) -> None:
    """Seed the local DB from healthy peers before joining the routing table.

    Calls GetPeers on the coordinator to get the list of donors and their
    assigned ring arcs, then opens a parallel Dump stream to each donor.
    Items are written into the local VectorStore as they arrive. Runs to
    completion before the heartbeat loop starts so the shard never enters
    the routing table with stale or empty data.

    Known limitations:
    - GetPeers is attempted once with a 10s timeout. If the coordinator is not
      yet up (e.g. Docker startup race), transfer is skipped entirely and the
      shard starts empty. No retry.
    - Individual Dump streams have no timeout. A stalled donor blocks that
      ThreadPoolExecutor worker — and therefore startup — indefinitely.
    - Writes that arrive on other shards during the dump are not captured.
      The gap is bounded by dump duration; the coordinator logs a warning on
      re-registration to flag this.
    """
    ctrl_channel = grpc.insecure_channel(COORDINATOR_HOST)
    ctrl_stub = vector_store_pb2_grpc.CoordinatorControlStub(ctrl_channel)

    try:
        resp = ctrl_stub.GetPeers(
            vector_store_pb2.GetPeersRequest(host=SHARD_HOST),
            timeout=10,
        )
    except grpc.RpcError as e:
        logger.warning(f"[StateTransfer] GetPeers failed ({e.details()}), skipping transfer")
        return

    if not resp.peers:
        logger.info("[StateTransfer] No peers registered yet — starting fresh")
        return

    logger.info(f"[StateTransfer] Pulling from {len(resp.peers)} donor(s)")

    def pull_arc(peer) -> int:
        shard_channel = grpc.insecure_channel(peer.host)
        shard_stub = vector_store_pb2_grpc.VectorStoreStub(shard_channel)
        count = 0
        try:
            for item in shard_stub.Dump(
                vector_store_pb2.DumpRequest(
                    start_hash=peer.start_hash, end_hash=peer.end_hash
                )
            ):
                servicer.service.add_item(item.id, item.text, list(item.embedding))
                count += 1
        except grpc.RpcError as e:
            logger.warning(
                f"[StateTransfer] Dump from {peer.host} failed ({e.details()}), "
                "arc may be incomplete"
            )
        logger.info(f"[StateTransfer] Received {count} items from {peer.host}")
        return count

    with futures.ThreadPoolExecutor(max_workers=len(resp.peers)) as pool:
        total = sum(pool.map(pull_arc, resp.peers))

    logger.info(
        f"[StateTransfer] Complete — {total} items received, "
        f"{servicer.service.vector_store.count()} items in local DB"
    )


def serve():
    servicer = VectorStoreServicer()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    vector_store_pb2_grpc.add_VectorStoreServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{SERVER_PORT}")
    logger.info(f"Shard started on port {SERVER_PORT}")
    # Start gRPC server before state transfer so this shard can serve Dump
    # requests to other recovering shards while seeding its own DB.
    server.start()

    if SKIP_STATE_TRANSFER:
        logger.info("[StateTransfer] Skipped (SKIP_STATE_TRANSFER=true)")
    else:
        _state_transfer(servicer)

    t = threading.Thread(target=_heartbeat_loop, daemon=True)
    t.start()
    logger.info(f"Heartbeat thread started -> {COORDINATOR_HOST} as {SHARD_HOST}")

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
