import numpy as np
from .vector_store import VectorStore


class VectorService:
    """High-level service for storing and searching pre-computed vector embeddings."""

    def __init__(self):
        """Initializes the service and its dependencies."""
        self.vector_store = VectorStore()

    def add_item(self, item_id: int, text: str, embedding: list[float]) -> None:
        """
        Stores a pre-computed embedding in the vector store.

        Args:
            item_id: The ID for the new entry.
            text: The original text (stored as metadata for result retrieval).
            embedding: The pre-computed embedding vector from the coordinator.
        """
        self.vector_store.upsert(item_id, text, np.array(embedding))

    def add_items_batch(self, items: list[tuple[int, str, list[float]]]) -> None:
        """
        Stores a batch of pre-computed embeddings in the vector store.

        Args:
            items: A list of (item_id, text, embedding) tuples.
        """
        for item_id, text, embedding in items:
            self.vector_store.upsert(item_id, text, np.array(embedding))

    def search(self, query_vector: list[float], top_k: int = 3) -> list[tuple[str, float]]:
        """
        Performs a similarity search against the stored vectors.

        Args:
            query_vector: The pre-computed query embedding from the coordinator.
            top_k: Number of most similar results to return.

        Returns:
            A list of (text, score) tuples for the top matches.
        """
        q = np.array(query_vector)
        ids = list(self.vector_store.store.keys())
        candidates = np.array(list(self.vector_store.store.values()))
        q_norm = q / np.linalg.norm(q)
        c_norms = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)
        sims = c_norms @ q_norm  # shape (n,)
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [
            (self.vector_store.metadata[ids[i]], float(sims[i]))
            for i in top_indices
        ]
