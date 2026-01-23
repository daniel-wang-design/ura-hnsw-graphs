from __future__ import annotations

from typing import Any
import numpy as np
import hnswlib

from interfaces.basic_database_interface import BasicHNSWIndex 


class StaticHNSWIndex(BasicHNSWIndex):
    """
    Static HNSW index:
      - create_index() builds from initial_vectors
      - insert() is a no-op (index not updated)
      - query() returns top-k neighbors from built index
    """

    def __init__(self) -> None:
        self._index: hnswlib.Index | None = None
        self._dim: int | None = None
        self._n: int = 0

    def create_index(self, initial_vectors: np.ndarray, **options: Any) -> None:
        x = np.asarray(initial_vectors, dtype=np.float32, order="C")
        if x.ndim != 2:
            raise ValueError(f"initial_vectors must be 2D (N,D); got shape {x.shape}")
        n, d = x.shape
        if n <= 0 or d <= 0:
            raise ValueError(f"invalid shape (N={n}, D={d})")

        # Common hnswlib options (with defaults)
        space = str(options.pop("space", "l2"))                 # "l2", "ip", "cosine"
        ef_construction = int(options.pop("ef_construction", 200))
        M = int(options.pop("M", 16))
        ef = int(options.pop("ef", 50))                        # query-time ef
        num_threads = int(options.pop("num_threads", -1))
        max_elements = options.pop("max_elements", n)
        if options:
            unknown = ", ".join(sorted(options.keys()))
            raise TypeError(f"Unknown option(s) for create_index: {unknown}")

        idx = hnswlib.Index(space=space, dim=d)
        idx.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)

        labels = np.arange(n, dtype=np.int64)                  # fixed IDs: 0..N-1
        idx.add_items(x, labels, num_threads=num_threads)
        idx.set_ef(ef)

        self._index = idx
        self._dim = d
        self._n = n

    def insert(self, vec_id: int, vec: np.ndarray) -> None:
        # No-op by design (static index)
        return

    def query(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._index is None or self._dim is None:
            raise RuntimeError("Index not built. Call create_index() first.")
        if not (1 <= k <= self._n):
            raise ValueError(f"k must be in [1, {self._n}], got {k}")

        q = np.asarray(q, dtype=np.float32, order="C")

        # Support single query (D,) or batch queries (Q,D)
        if q.ndim == 1:
            if q.shape[0] != self._dim:
                raise ValueError(f"dim mismatch: expected {self._dim}, got {q.shape[0]}")
            labels, dists = self._index.knn_query(q.reshape(1, self._dim), k=k)
            return labels[0], dists[0]

        if q.ndim == 2:
            if q.shape[1] != self._dim:
                raise ValueError(f"dim mismatch: expected {self._dim}, got {q.shape[1]}")
            labels, dists = self._index.knn_query(q, k=k)
            return labels, dists

        raise ValueError(f"q must be 1D (D,) or 2D (Q,D); got shape {q.shape}")
