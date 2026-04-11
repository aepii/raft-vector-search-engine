from typing import Union, List
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wrapper for the SentenceTransformer model to handle vector operations."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initializes the transformer model."""
        self.model = SentenceTransformer(model_name)

    def encode(self, text: Union[str, list[str]]) -> np.ndarray:
        """
        Converts text strings into embeddings.

        Args:
            text: A raw string or a list of raw strings to encode.

        Returns:
            A numpy array representing the embedding(s).
        """
        return self.model.encode(text)

