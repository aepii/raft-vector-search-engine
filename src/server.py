import grpc
from concurrent import futures
import vector_store_pb2, vector_store_pb2_grpc
from classes.vector_service import VectorService


class VectorStoreServicer(vector_store_pb2_grpc.VectorStoreServicer):
    def __init__(self):
        self.service = VectorService()

    def Upsert(self, request, context):
        self.service.add_item(request.id, request.text)
        return vector_store_pb2.UpsertResponse(
            upsert_status=f"ID {request.id} indexed."
        )

    def Search(self, request, context):
        results = self.service.search(request.query_text, top_k=request.top_k)
        return vector_store_pb2.SearchResponse(results=results)


def serve():
    print("Starting Vector Store Server...")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    vector_store_pb2_grpc.add_VectorStoreServicer_to_server(
        VectorStoreServicer(), server
    )
    server.add_insecure_port("[::]:50051")
    print("Vector Store Server started on port 50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
