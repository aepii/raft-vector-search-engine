import hashlib
import bisect
from typing import Optional


class ConsistentHashRing:
    """
    Consistent hash ring using SHA-256 and virtual nodes.

    Each physical host gets `virtual_nodes` positions on the ring.
    A key is mapped to the first host clockwise from its hash position.
    """

    def __init__(self, virtual_nodes: int = 150):
        self._virtual_nodes = virtual_nodes
        self._ring: list[tuple[int, str]] = []  # sorted by hash point
        self._points: list[int] = []            # parallel list for bisect

    def _hash(self, key: str) -> int:
        return int(hashlib.sha256(key.encode()).hexdigest(), 16)

    def add_node(self, host: str) -> None:
        for i in range(self._virtual_nodes):
            h = self._hash(f"{host}#vnode{i}")
            pos = bisect.bisect_left(self._points, h)
            self._points.insert(pos, h)
            self._ring.insert(pos, (h, host))

    def remove_node(self, host: str) -> None:
        filtered = [(h, n) for h, n in self._ring if n != host]
        self._ring = filtered
        self._points = [h for h, _ in filtered]

    def get_node(self, key: str) -> Optional[str]:
        if not self._ring:
            return None
        h = self._hash(key)
        pos = bisect.bisect_left(self._points, h) % len(self._points)
        return self._ring[pos][1]

    def nodes(self) -> list[str]:
        return list({host for _, host in self._ring})

    def __len__(self) -> int:
        return len(self.nodes())
