import os
import grpc
from concurrent import futures
import vector_store_pb2, vector_store_pb2_grpc
from classes.vector_service import VectorService
from utils.logger import get_logger
from dotenv import load_dotenv
import torch

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

load_dotenv()

SERVER_PORT = os.getenv("SERVER_PORT", "50051")

logger = get_logger(f"SHARD:{SERVER_PORT}")


class VectorStoreServicer(vector_store_pb2_grpc.VectorStoreServicer):
    """gRPC servicer implementation for the VectorStore service."""

    def __init__(self):
        self.service = VectorService()

    def Upsert(self, request, context):
        """
        Inserts or updates an item in the vector store.
        """
        trace_id = request.trace_id
        item = request.item

        logger.info(f"[{trace_id}] [Upsert] id={item.id}")
        self.service.add_item(item.id, item.text)
        logger.info(f"[{trace_id}] [Upsert] id={item.id} indexed")

        return vector_store_pb2.UpsertResponse(status=f"ID {request.id} indexed.")

    def UpsertBatch(self, request, context):
        """
        Inserts or updates a batch of items in the vector store.
        """
        trace_id = request.trace_id

        logger.info(f"[{trace_id}] [UpsertBatch] size={len(request.items)}")

        items = [(item.id, item.text) for item in request.items]
        self.service.add_items_batch(items)

        logger.info(f"[{trace_id}] [UpsertBatch] indexed {len(items)} items")

        statuses = [f"ID {item.id} indexed." for item in request.items]
        return vector_store_pb2.UpsertBatchResponse(statuses=statuses)

    def Count(self, request, context):
        count = len(self.service.vector_store.store)
        return vector_store_pb2.CountResponse(count=count)

    def Search(self, request, context):
        """
        Performs a similarity search over the indexed vectors.
        """
        trace_id = request.trace_id

        logger.info(f"[{trace_id}] [Search] query='{request.query_text}'")

        results = self.service.search(request.query_text, top_k=request.top_k)

        logger.info(f"[{trace_id}] [Search] returning {len(results)} results")

        return vector_store_pb2.SearchResponse(
            results=[
                vector_store_pb2.SearchResult(text=text, score=score)
                for text, score in results
            ]
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    vector_store_pb2_grpc.add_VectorStoreServicer_to_server(
        VectorStoreServicer(), server
    )
    server.add_insecure_port(f"[::]:{SERVER_PORT}")
    logger.info(f"Shard started on port {SERVER_PORT}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
