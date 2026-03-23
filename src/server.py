import os
import grpc
from concurrent import futures
import vector_store_pb2, vector_store_pb2_grpc
from classes.vector_service import VectorService
from dotenv import load_dotenv

load_dotenv()

SERVER_PORT = os.getenv("SERVER_PORT", "50051")


class VectorStoreServicer(vector_store_pb2_grpc.VectorStoreServicer):
    """gRPC servicer implementation for the VectorStore service."""

    def __init__(self):
        self.service = VectorService()

    def Upsert(self, request, context):
        """
        Inserts or updates an item in the vector store.
        """
        self.service.add_item(request.id, request.text)
        return vector_store_pb2.UpsertResponse(
            upsert_status=f"ID {request.id} indexed."
        )

    def Search(self, request, context):
        """
        Performs a similarity search over the indexed vectors.
        """
        results = self.service.search(request.query_text, top_k=request.top_k)
        return vector_store_pb2.SearchResponse(
            results=[
                vector_store_pb2.SearchResult(text=text, score=score)
                for text, score in results
            ]
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor())
    vector_store_pb2_grpc.add_VectorStoreServicer_to_server(
        VectorStoreServicer(), server
    )
    server.add_insecure_port(f"[::]:{SERVER_PORT}")
    print(f"Vector Store Server started on port {SERVER_PORT}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
