import os
import random
import grpc
import vector_store_pb2, vector_store_pb2_grpc
from utils.logger import get_logger, new_trace_id
from dotenv import load_dotenv

load_dotenv()

COORDINATOR_HOST = os.getenv("COORDINATOR_HOST", "localhost:50051")

logger = get_logger("CLIENT")


class VectorStoreClient:
    """Client SDK for interacting with the vector store via gRPC."""

    def __init__(self, host: str = None):
        """Initializes the client."""
        self.host = host or COORDINATOR_HOST
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
        trace_id = new_trace_id("upsert")
        logger.info(f"[{trace_id}] [upsert] id={item_id}")

        request = vector_store_pb2.UpsertRequest(
            trace_id=trace_id, item=vector_store_pb2.UpsertItem(id=item_id, text=text)
        )
        response = self.stub.Upsert(request)

        logger.info(f"[{trace_id}] response: {response.status}")
        return response.status

    def upsert_batch(self, items: list[tuple[int, str]]) -> list[str]:
        """
        Inserts or updates a batch of items in the remote vector store.

        Args:
            items: A list of item where each item has an item_id and text.

        Returns:
            A list of status messages from the server indicating success or failure.
        """
        trace_id = new_trace_id("batch")
        logger.info(f"[{trace_id}] [upsert_batch] size={len(items)}")

        request_items = [
            vector_store_pb2.UpsertItem(id=item_id, text=text)
            for item_id, text in items
        ]
        request = vector_store_pb2.UpsertBatchRequest(
            trace_id=trace_id, items=request_items
        )
        response = self.stub.UpsertBatch(request)

        logger.info(f"[{trace_id}] batch complete, {len(response.statuses)} statuses")
        return list(response.statuses)

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """
        Performs a semantic search against the remote vector store.

        Args:
            query: Query string to search for.
            top_k: Number of top matching results to return.

        Returns:
            A list of matching text results ranked by similarity.
        """
        trace_id = new_trace_id("search")
        logger.info(f"[{trace_id}] [search] query='{query}' top_k={top_k}")

        request = vector_store_pb2.SearchRequest(
            trace_id=trace_id, query_text=query, top_k=top_k
        )
        response = self.stub.Search(request)
        results = [(result.text, result.score) for result in response.results]

        logger.info(f"[{trace_id}] received {len(results)} results")
        return [(result.text, result.score) for result in response.results]

    def close(self):
        self.channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
