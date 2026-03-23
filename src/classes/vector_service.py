from typing import List
import torch
import numpy as np
from .embedding_model import EmbeddingModel
from .vector_store import VectorStore


class VectorService:
    """High-level service for embedding text and performing semantic search over stored items."""

    def __init__(self):
        """Initializes the service and its dependencies."""
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()

    def add_item(self, item_id: int, text: str) -> None:
        """
        Encodes text and stores it in the vector store.

        Args:
            item_id: The ID for the new entry.
            text: The text to be indexed.
        """
        query_embedding = self.embedding_model.encode(text)
        self.vector_store.upsert(item_id, text, query_embedding)

    def search(self, text: str, top_k: int = 3) -> list[tuple[str, float]]:
        """
        Performs a semantic search against the stored vectors.

        Args:
            text: The query string.
            top_k: Number of most similar results to return.

        Returns:
            A list of strings representing the top matches.
        """
        # Encode the query string
        query_embedding = self.embedding_model.encode(text)
        # Extract ids from the store
        ids = list(self.vector_store.store.keys())
        # Convert the store's values into a matrix
        candidate_embeddings = np.array(list(self.vector_store.store.values()))
        # Calculate similarities
        similarities = self.embedding_model.similarity(
            query_embedding, candidate_embeddings
        )
        # Get the top indices
        top_results = torch.argsort(similarities, descending=True)[0][:top_k]

        # Return the mapped indices back to metadata
        return [
            (self.vector_store.metadata[ids[i]], float(similarities[0][i]))
            for i in top_results
        ]
