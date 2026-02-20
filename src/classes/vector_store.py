import numpy as np

# 
class VectorStore:
    def __init__(self):
        self.store = {}  # ID -> Vector Embedding
        self.metadata = {}  # ID -> Metadata

    def upsert(self, item_id: int, text: str, embedding: np.ndarray):
        self.metadata[item_id] = text
        self.store[item_id] = np.array(embedding)
