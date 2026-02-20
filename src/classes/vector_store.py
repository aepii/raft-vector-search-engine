from typing import Dict, List
import numpy as np


class VectorStore:
    """An in-memory storage for vector embeddings and their metadata."""

    def __init__(self) -> None:
        """Initializes the vector store."""
        # Maps an item_id to the vector embedding
        self.store: Dict[int, np.ndarray] = {}
        # Maps an item_id to the original text content
        self.metadata = {}

    def upsert(self, item_id: int, text: str, embedding: np.ndarray) -> None:
        """
        Inserts or updates a vector and its metadata.

        Args:
            item_id: Unique identifier for the item.
            text: The raw string content.
            embedding: The numerical representation of the text.
        """
        self.metadata[item_id] = text
        self.store[item_id] = np.array(embedding)
