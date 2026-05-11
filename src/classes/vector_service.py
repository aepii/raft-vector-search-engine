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
        self._ids_cache = []
        self._embeddings_cache = None

    def add_item(self, item_id: int, text: str) -> None:
        """
        Encodes text and stores it in the vector store.

        Args:
            item_id: The ID for the new entry.
            text: The text to be indexed.
        """
        query_embedding = self.embedding_model.encode(text)
        self.vector_store.upsert(item_id, text, query_embedding)
        self._update_cache()

    def add_items_batch(self, items: list[tuple[int, str]]) -> None:
        """
        Encodes batch of items and stores it in the vector store.

        Args:
            items: A list of item where each item has an item_id and text.
        """
        texts = [text for _, text in items]
        query_embeddings = self.embedding_model.encode(texts)
        for (item_id, text), query_embedding in zip(items, query_embeddings):
            self.vector_store.upsert(item_id, text, query_embedding)
        self._update_cache()

    def _update_cache(self):
        self._ids_cache = list(self.vector_store.store.keys())
        self._embeddings_cache = np.array(list(self.vector_store.store.values()))

    def search(self, text: str, top_k: int = 3) -> list[tuple[str, float]]:
        """
        Performs a semantic search against the stored vectors.

        Args:
            text: The query string.
            top_k: Number of most similar results to return.

        Returns:
            A list of strings representing the top matches.
        """
        if self._embeddings_cache is None or len(self._ids_cache) == 0:
            return []

        # Encode the query string
        query_embedding = self.embedding_model.encode(text)

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

        return [
            (self.vector_store.metadata[self._ids_cache[i]], float(similarities[i]))
            for i in top_indices
        ]
