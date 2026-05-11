import numpy as np
from .vector_store import VectorStore


class VectorService:
    """High-level service for storing and searching pre-computed vector embeddings."""

    def __init__(self, vector_store: VectorStore):
        """
        Initializes the service with the provided vector store.

        Args:
            vector_store: The VectorStore instance to use for storage and search.
        """
        self.vector_store = vector_store
        self._ids_cache = []
        self._texts_cache = []
        self._embeddings_cache = None

    def add_item(self, item_id: int, text: str, embedding: list[float]) -> None:
        """
        Stores a pre-computed embedding in the vector store.

        Args:
            item_id: The ID for the new entry.
            text: The original text (stored as metadata for result retrieval).
            embedding: The pre-computed embedding vector from the coordinator.
        """
        self.vector_store.upsert(item_id, text, embedding)
        self._update_cache()

    def add_items_batch(self, items: list[tuple[int, str, list[float]]]) -> None:
        """
        Stores a batch of pre-computed embeddings in the vector store.

        Args:
            items: A list of (item_id, text, embedding) tuples.
        """
        for item_id, text, embedding in items:
            self.vector_store.upsert(item_id, text, embedding)
        self._update_cache()

    def _update_cache(self):
        rows = self.vector_store.scan()  # returns list of (id, text, embedding)
        self._ids_cache = [row[0] for row in rows]
        self._texts_cache = [row[1] for row in rows]
        self._embeddings_cache = np.array([row[2] for row in rows])

    def search(
        self, query_vector: list[float], top_k: int = 3
    ) -> list[tuple[str, float]]:
        """
        Performs a similarity search against the stored vectors.

        Args:
            query_vector: The pre-computed query embedding from the coordinator.
            top_k: Number of most similar results to return.

        Returns:
            A list of (text, score) tuples for the top matches.
        """
        if self._embeddings_cache is None or len(self._ids_cache) == 0:
            return []

        # Encode the query string
        query_embedding = np.array(query_vector)

        dot = np.dot(self._embeddings_cache, query_embedding.T)
        norm_store = np.linalg.norm(self._embeddings_cache, axis=1)
        norm_query = np.linalg.norm(query_embedding)
        similarities = dot / (norm_store * norm_query + 1e-10)

        # Top-k efficiently
        if top_k >= len(similarities):
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(-similarities, top_k)[:top_k]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        return [(self._texts_cache[i], float(similarities[i])) for i in top_indices]
