from __future__ import annotations

from typing import Any
import numpy as np
import hnsw_cpp

from interfaces.basic_database_interface import BasicHNSWIndex


class PybindHNSWIndex(BasicHNSWIndex):
    """
    HNSW index backed by the custom C++ pybind module.
    Supports:
      - create_index(): initialize and bulk insert vectors
      - insert(): insert one vector
      - query(): single-query top-k search
    """

    def __init__(self) -> None:
        self._index: hnsw_cpp.HNSW | None = None
        self._dim: int | None = None
        self._n: int = 0

    def create_index(self, initial_vectors: np.ndarray, **options: Any) -> None:
        x = np.asarray(initial_vectors, dtype=np.float64, order="C")
        if x.ndim != 2:
            raise ValueError(f"initial_vectors must be 2D (N,D); got shape {x.shape}")

        n, d = x.shape
        if n <= 0 or d <= 0:
            raise ValueError(f"invalid shape (N={n}, D={d})")

        M = int(options.pop("M", 16))
        ef_construction = int(options.pop("ef_construction", 200))
        random_seed = int(options.pop("random_seed", 42))

        # query-time default ef to use later
        self._default_ef = int(options.pop("ef", max(50, 10)))
        if options:
            unknown = ", ".join(sorted(options.keys()))
            print(f"Unknown option(s) for create_index: {unknown}")

        idx = hnsw_cpp.HNSW()
        idx.init(dim=d, M=M, ef_construction=ef_construction, random_seed=random_seed)

        # Bulk build by repeated insert
        for i in range(n):
            idx.insert(i, x[i].tolist())

        self._index = idx
        self._dim = d
        self._n = n

    def insert(self, vec_id: int, vec: np.ndarray) -> None:
        if self._index is None or self._dim is None:
            raise RuntimeError("Index not built. Call create_index() first.")

        vec = np.asarray(vec, dtype=np.float64, order="C")
        if vec.ndim != 1:
            raise ValueError(f"vec must be 1D (D,); got shape {vec.shape}")
        if vec.shape[0] != self._dim:
            raise ValueError(f"dim mismatch: expected {self._dim}, got {vec.shape[0]}")

        self._index.insert(int(vec_id), vec.tolist())
        self._n += 1

    def query(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._index is None or self._dim is None:
            raise RuntimeError("Index not built. Call create_index() first.")
        if not (1 <= k <= self._n):
            raise ValueError(f"k must be in [1, {self._n}], got {k}")

        q = np.asarray(q, dtype=np.float64, order="C")

        # Single query only, because current pybind binding only supports one query vector
        if q.ndim != 1:
            raise ValueError(f"q must be 1D (D,); got shape {q.shape}")
        if q.shape[0] != self._dim:
            raise ValueError(f"dim mismatch: expected {self._dim}, got {q.shape[0]}")

        result = self._index.knn_search(q.tolist(), k, self._default_ef)
        labels = np.array([label for label, _dist in result], dtype=np.int64)
        dists = np.array([dist for _label, dist in result], dtype=np.float64)
        return labels, dists