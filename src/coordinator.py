import os
import grpc
from concurrent import futures
import vector_store_pb2, vector_store_pb2_grpc
from dotenv import load_dotenv

load_dotenv()

SHARD_HOSTS = os.getenv(
    "SHARD_HOSTS", "localhost:50051,localhost:50052,localhost:50053"
).split(",")
COORDINATOR_PORT = os.getenv("COORDINATOR_PORT", "50050")


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
        shard_index = self._route(request.id)
        print(
            f"Upsert ID {request.id} - shard {shard_index} ({self.shard_hosts[shard_index]})"
        )
        response = self.stubs[shard_index].Upsert(request)

        return vector_store_pb2.UpsertResponse(
            upsert_status=f"[shard {shard_index}] {response.upsert_status}"
        )

    def Search(self, request, context):
        def query_shard(stub, shard_index):
            try:
                response = stub.Search(request)
                return list(response.results)
            except grpc.RpcError as e:
                print(f"Shard {shard_index} unavailable: {e.details()}")
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
        return vector_store_pb2.SearchResponse(results=all_results[: request.top_k])


def serve():
    server = grpc.server(futures.ThreadPoolExecutor())
    vector_store_pb2_grpc.add_VectorStoreServicer_to_server(
        CoordinatorServicer(SHARD_HOSTS), server
    )
    server.add_insecure_port(f"[::]:{COORDINATOR_PORT}")
    print(f"Vector Store Server started on port {COORDINATOR_PORT}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
