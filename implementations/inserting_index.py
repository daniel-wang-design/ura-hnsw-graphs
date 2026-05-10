from __future__ import annotations

from typing import Any
import numpy as np
import hnswlib

from interfaces.basic_database_interface import BasicHNSWIndex


class UpdatingHNSWIndex(BasicHNSWIndex):
    """
    HNSW index:
      - create_index() builds from initial_vectors
      - insert() adds one new vector to the existing index
      - query() returns top-k neighbors from built index
    """

    def __init__(self) -> None:
        self._index: hnswlib.Index | None = None
        self._dim: int | None = None
        self._n: int = 0
        self._max_elements: int = 0
        self._ids: set[int] = set()
        self._num_threads: int = -1

    def create_index(self, initial_vectors: np.ndarray, **options: Any) -> None:
        x = np.asarray(initial_vectors, dtype=np.float32, order="C")
        if x.ndim != 2:
            raise ValueError(f"initial_vectors must be 2D (N,D); got shape {x.shape}")

        n, d = x.shape
        if n <= 0 or d <= 0:
            raise ValueError(f"invalid shape (N={n}, D={d})")

        # Common hnswlib options
        space = str(options.pop("space", "l2"))                 # "l2", "ip", "cosine"
        ef_construction = int(options.pop("ef_construction", 200))
        M = int(options.pop("M", 16))
        ef = int(options.pop("ef", 50))                        # query-time ef
        num_threads = int(options.pop("num_threads", -1))
        max_elements = int(options.pop("max_elements", n))

        if max_elements < n:
            raise ValueError(f"max_elements ({max_elements}) must be >= initial size ({n})")

        if options:
            unknown = ", ".join(sorted(options.keys()))
            print(f"Unknown option(s) for create_index: {unknown}")

        idx = hnswlib.Index(space=space, dim=d)
        idx.init_index(
            max_elements=max_elements,
            ef_construction=ef_construction,
            M=M,
        )

        labels = np.arange(n, dtype=np.int64)
        idx.add_items(x, labels, num_threads=num_threads)
        idx.set_ef(ef)

        self._index = idx
        self._dim = d
        self._n = n
        self._max_elements = max_elements
        self._num_threads = num_threads
        self._ids = set(map(int, labels.tolist()))

    def insert(self, vec_id: int, vec: np.ndarray) -> None:
        if self._index is None or self._dim is None:
            raise RuntimeError("Index not built. Call create_index() first.")

        vec_id = int(vec_id)
        if vec_id in self._ids:
            raise ValueError(f"vec_id {vec_id} already exists in the index")

        x = np.asarray(vec, dtype=np.float32, order="C")
        if x.ndim != 1:
            raise ValueError(f"vec must be 1D (D,), got shape {x.shape}")
        if x.shape[0] != self._dim:
            raise ValueError(f"dim mismatch: expected {self._dim}, got {x.shape[0]}")

        # Grow capacity if needed
        if self._n >= self._max_elements:
            new_capacity = max(self._max_elements * 2, self._n + 1)
            self._index.resize_index(new_capacity)
            self._max_elements = new_capacity

        self._index.add_items(
            x.reshape(1, self._dim),
            np.asarray([vec_id], dtype=np.int64),
            num_threads=self._num_threads,
        )

        self._ids.add(vec_id)
        self._n += 1

    def query(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._index is None or self._dim is None:
            raise RuntimeError("Index not built. Call create_index() first.")
        if not (1 <= k <= self._n):
            raise ValueError(f"k must be in [1, {self._n}], got {k}")

        q = np.asarray(q, dtype=np.float32, order="C")

        # Single query: (D,)
        if q.ndim == 1:
            if q.shape[0] != self._dim:
                raise ValueError(f"dim mismatch: expected {self._dim}, got {q.shape[0]}")
            labels, dists = self._index.knn_query(q.reshape(1, self._dim), k=k)
            return labels[0], dists[0]

        # Batch query: (Q,D)
        if q.ndim == 2:
            if q.shape[1] != self._dim:
                raise ValueError(f"dim mismatch: expected {self._dim}, got {q.shape[1]}")
            labels, dists = self._index.knn_query(q, k=k)
            return labels, dists

        raise ValueError(f"q must be 1D (D,) or 2D (Q,D); got shape {q.shape}")