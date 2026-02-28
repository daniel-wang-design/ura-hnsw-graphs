from __future__ import annotations

from typing import Any
import numpy as np

from interfaces.basic_database_interface import BasicHNSWIndex
from hnsw.hnsw import SimpleHNSWIndex


class StaticHNSWIndex(BasicHNSWIndex):
    """
    Static (pure-python) HNSW index:
      - create_index() builds from initial_vectors
      - insert() is a no-op (index not updated)
      - query() returns top-k neighbors from built index

    Notes:
      - SimpleHNSWIndex supports only space='l2' (squared L2 distances).
      - knn_query returns squared L2 distances (same convention as many ANN libs).
    """

    def __init__(self) -> None:
        self._index: SimpleHNSWIndex | None = None
        self._dim: int | None = None
        self._n: int = 0

        self._space: str = "l2"
        self._query_ef: int = 50

    def create_index(self, initial_vectors: np.ndarray, **options: Any) -> None:
        x = np.asarray(initial_vectors, dtype=np.float32, order="C")
        if x.ndim != 2:
            raise ValueError(f"initial_vectors must be 2D (N,D); got shape {x.shape}")

        n, d = x.shape
        if n <= 0 or d <= 0:
            raise ValueError(f"invalid shape (N={n}, D={d})")

        # hnswlib-like options
        space = str(options.pop("space", "l2"))
        ef_construction = int(options.pop("ef_construction", 200))
        M = int(options.pop("M", 16))
        ef = int(options.pop("ef", 50))  # query ef (ef_search)

        # Optional extras for this implementation
        random_seed = int(options.pop("random_seed", 42))
        select = str(options.pop("select", "heuristic"))  # "heuristic" or "simple"
        max_elements = int(options.pop("max_elements", n))

        _num_threads = options.pop("num_threads", None)  # ignored (single-thread pure python)

        if options:
            unknown = ", ".join(sorted(options.keys()))
            raise TypeError(f"Unknown option(s) for create_index: {unknown}")

        if space != "l2":
            raise ValueError(
                f"space={space!r} not supported by SimpleHNSWIndex (only 'l2'). "
                f"If you want cosine later, we can add normalization + dot-based distance."
            )

        idx = SimpleHNSWIndex(space="l2", dim=d)
        idx.init_index(
            max_elements=max_elements,
            ef_construction=ef_construction,
            M=M,
            random_seed=random_seed,
            ef_search=ef,
            select=select,
        )

        # IDs become 0..N-1 by insertion order (like your hnswlib labels)
        ids = np.arange(n, dtype=np.int64)
        idx.add_items(x, ids=ids)

        self._index = idx
        self._dim = d
        self._n = n
        self._space = space
        self._query_ef = ef

    def insert(self, vec_id: int, vec: np.ndarray) -> None:
        # No-op by design (static index)
        return

        # If you later want dynamic inserts:
        # if self._index is None or self._dim is None:
        #     raise RuntimeError("Index not built. Call create_index() first.")
        # v = np.asarray(vec, dtype=np.float32, order="C")
        # if v.ndim != 1 or v.shape[0] != self._dim:
        #     raise ValueError(f"dim mismatch: expected {self._dim}, got {v.shape}")
        # self._index.add_items(v, ids=[int(vec_id)])
        # self._n += 1

    def query(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._index is None or self._dim is None:
            raise RuntimeError("Index not built. Call create_index() first.")
        if not (1 <= k <= self._n):
            raise ValueError(f"k must be in [1, {self._n}], got {k}")

        # Ensure query ef is applied (SimpleHNSWIndex uses ef_search internally)
        self._index.set_ef(self._query_ef)

        q = np.asarray(q, dtype=np.float32, order="C")

        # Single query (D,)
        if q.ndim == 1:
            if q.shape[0] != self._dim:
                raise ValueError(f"dim mismatch: expected {self._dim}, got {q.shape[0]}")
            labels2d, dists2d = self._index.knn_query(q, k=k)  # shapes (1,k)
            return labels2d[0], dists2d[0]

        # Batch queries (Q, D)
        if q.ndim == 2:
            if q.shape[1] != self._dim:
                raise ValueError(f"dim mismatch: expected {self._dim}, got {q.shape[1]}")
            labels2d, dists2d = self._index.knn_query(q, k=k)  # shapes (Q,k)
            return labels2d, dists2d

        raise ValueError(f"q must be 1D (D,) or 2D (Q,D); got shape {q.shape}")