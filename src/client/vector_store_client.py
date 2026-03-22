import os
import random
import grpc
import vector_store_pb2, vector_store_pb2_grpc
from dotenv import load_dotenv

load_dotenv()


class VectorStoreClient:
    """Client SDK for interacting with the vector store via gRPC."""

    def __init__(self, host: str = None):
        """Initializes the client."""
        self.host = host or os.getenv("SERVER_HOST", "localhost:50051")
        self.channel = grpc.insecure_channel(self.host)
        self.stub = vector_store_pb2_grpc.VectorStoreStub(self.channel)

    def upsert(self, item_id: int, text: str) -> str:
        """
        Inserts or updates an item in the remote vector store.

        Args:
            item_id: Unique identifier for the item.
            text: Raw text to be stored and embedded by the server.

        Returns:
            A status message from the server indicating success or failure.
        """
        request = vector_store_pb2.UpsertRequest(id=item_id, text=text)
        response = self.stub.Upsert(request)
        return response.upsert_status

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """
        Performs a semantic search against the remote vector store.

        Args:
            query: Query string to search for.
            top_k: Number of top matching results to return.

        Returns:
            A list of matching text results ranked by similarity.
        """
        request = vector_store_pb2.SearchRequest(query_text=query, top_k=top_k)
        response = self.stub.Search(request)
        return list(response.results)

    def close(self):
        self.channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
