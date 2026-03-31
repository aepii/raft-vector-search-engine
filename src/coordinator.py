import os
import time
import grpc
from concurrent import futures
import vector_store_pb2, vector_store_pb2_grpc
from utils.logger import get_logger
from dotenv import load_dotenv

load_dotenv()

SHARD_HOSTS = os.getenv(
    "SHARD_HOSTS", "localhost:50051,localhost:50052,localhost:50053"
).split(",")
COORDINATOR_PORT = os.getenv("COORDINATOR_PORT", "50050")

logger = get_logger("COORDINATOR")


class CoordinatorServicer(vector_store_pb2_grpc.VectorStoreServicer):
    def __init__(self, shard_hosts: list[str]):
        self.shard_hosts = shard_hosts
        self.num_shards = len(shard_hosts)

        self.stubs = [
            vector_store_pb2_grpc.VectorStoreStub(grpc.insecure_channel(host))
            for host in shard_hosts
        ]

        print(f"Coordinator ready - {self.num_shards} shards: {self.shard_hosts}")

    def _route(self, item_id: int) -> int:
        """Modulo hash"""
        return item_id % self.num_shards

    def Upsert(self, request, context):
        trace_id = request.trace_id
        item = request.item
        shard_index = self._route(item.id)

        logger.info(
            f"[{trace_id}] [Upsert] id={item.id} -> shard {shard_index} ({self.shard_hosts[shard_index]})"
        )

        response = self.stubs[shard_index].Upsert(request)

        logger.info(f"[{trace_id}] [Upsert] id={item.id} complete: {response.status}")

        return vector_store_pb2.UpsertResponse(
            status=f"[shard {shard_index}] {response.upsert_status}"
        )

    def UpsertBatch(self, request, context):
        trace_id = request.trace_id

        logger.info(f"[{trace_id}] [UpsertBatch] received size={len(request.items)}")

        shard_batches = {i: [] for i in range(self.num_shards)}
        for item in request.items:
            shard_index = self._route(item.id)
            logger.info(f"[{trace_id}] item id={item.id} -> shard {shard_index}")
            shard_batches[shard_index].append(item)

        responses = []
        for shard_index, shard_batch in shard_batches.items():
            if not shard_batch:
                continue

            logger.info(
                f"[{trace_id}] sending {len(shard_batch)} items to shard {shard_index} ({self.shard_hosts[shard_index]})"
            )

            batch_request = vector_store_pb2.UpsertBatchRequest(items=shard_batch)

            start = time.perf_counter()
            response = self.stubs[shard_index].UpsertBatch(batch_request)
            elapsed = (time.perf_counter() - start) * 1000

            logger.info(
                f"[{trace_id}] shard {shard_index} batch complete in {elapsed:.2f}ms"
            )

            responses.extend(response.statuses)

        logger.info(f"[{trace_id}] [UpsertBatch] done, {len(responses)} total statuses")

        return vector_store_pb2.UpsertBatchResponse(statuses=responses)

    def Search(self, request, context):
        trace_id = request.trace_id

        logger.info(
            f"[{trace_id}] [Search] query='{request.query_text[:40]}' top_k={request.top_k}"
        )

        def query_shard(stub, shard_index):
            try:
                start = time.perf_counter()
                response = stub.Search(request)
                elapsed = (time.perf_counter() - start) * 1000

                logger.info(
                    f"[{trace_id}] shard {shard_index} returned {len(response.results)} results in {elapsed:.2f}ms"
                )
                return list(response.results)
            except grpc.RpcError as e:
                logger.error(
                    f"[{trace_id}] shard {shard_index} unavailable: {e.details()}"
                )
                return []

        all_results = []
        with futures.ThreadPoolExecutor(max_workers=self.num_shards) as executor:
            future_to_shard = {
                executor.submit(query_shard, stub, i): i
                for i, stub in enumerate(self.stubs)
            }
            for future in futures.as_completed(future_to_shard):
                all_results.extend(future.result())

        all_results.sort(key=lambda x: x.score, reverse=True)
        top = all_results[: request.top_k]

        logger.info(f"[{trace_id}] Search merged -> returning top {len(top)} results")
        return vector_store_pb2.SearchResponse(results=top)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    vector_store_pb2_grpc.add_VectorStoreServicer_to_server(
        CoordinatorServicer(SHARD_HOSTS), server
    )
    server.add_insecure_port(f"[::]:{COORDINATOR_PORT}")
    logger.info(f"Coordinator started on port {COORDINATOR_PORT}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
