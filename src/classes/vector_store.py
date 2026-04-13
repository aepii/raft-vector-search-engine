import sqlite3
import struct
import threading

import sqlite_vec

# Output dimension of all-MiniLM-L6-v2, the embedding model used by the coordinator.
EMBEDDING_DIM = 384


class VectorStore:
    """sqlite-vec-backed storage for vector embeddings and their metadata."""

    def __init__(self, db_path: str = ":memory:", dim: int = EMBEDDING_DIM) -> None:
        """
        Initializes the vector store.

        Args:
            db_path: Path to the SQLite database file. Defaults to an in-memory
                database, which is useful for tests. Pass a file path for
                persistent storage.
            dim: Embedding dimension. Defaults to EMBEDDING_DIM (384). Override
                in tests when using smaller synthetic vectors.
        """
        self.dim = dim
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self._create_tables()

    def _create_tables(self) -> None:
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS items (
                    id        INTEGER PRIMARY KEY,
                    text      TEXT    NOT NULL,
                    upsert_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                )
            """)
            self.conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_items
                    USING vec0(embedding float[{self.dim}] distance_metric=cosine)
            """)

    def upsert(self, item_id: int, text: str, embedding: list) -> None:
        """
        Inserts or updates a vector and its metadata atomically.

        Args:
            item_id: Unique identifier for the item.
            text: The raw string content.
            embedding: The numerical representation of the text as a list of floats.
        """
        packed = struct.pack(f"{self.dim}f", *embedding)
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO items(id, text) VALUES (?, ?)"
                " ON CONFLICT(id) DO UPDATE SET text=excluded.text,"
                " upsert_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
                (item_id, text),
            )
            self.conn.execute("DELETE FROM vec_items WHERE rowid = ?", (item_id,))
            self.conn.execute(
                "INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)",
                (item_id, packed),
            )

    def search(self, embedding: list, top_k: int = 3) -> list[tuple[str, float]]:
        """
        Returns the top_k most similar items to the given embedding.

        Args:
            embedding: Query vector as a list of floats.
            top_k: Number of results to return.

        Returns:
            A list of (text, score) tuples where score is cosine similarity
            (1.0 = identical).
        """
        packed = struct.pack(f"{self.dim}f", *embedding)
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT i.text, v.distance
                FROM (
                    SELECT rowid, distance
                    FROM vec_items
                    WHERE embedding MATCH ?
                    LIMIT ?
                ) v
                JOIN items i ON i.id = v.rowid
                ORDER BY v.distance
                """,
                (packed, top_k),
            ).fetchall()
        return [(text, 1.0 - distance) for text, distance in rows]

    def count(self) -> int:
        """Returns the number of items currently stored."""
        with self._lock:
            row = self.conn.execute("SELECT COUNT(*) FROM items").fetchone()
        return row[0]
