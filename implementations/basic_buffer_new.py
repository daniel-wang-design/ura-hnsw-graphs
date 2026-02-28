from __future__ import annotations

from typing import Any
import numpy as np
from hnsw.hnsw import HNSW

from interfaces.basic_database_interface import BasicHNSWIndex

# IMPORTANT: import or include the HNSW class you pasted earlier.
# If it's in the same file, you don't need this import.
# from your_module import HNSW


class StaticBufferHNSWIndex(BasicHNSWIndex):
    """
    Static (pure-python) HNSW index + buffered inserts:
      - create_index() builds from initial_vectors (static)
      - insert() buffers vectors (no index update)
      - query() gets top-k from HNSW, then replaces with better buffered vectors if any
    """

    def __init__(self) -> None:
        self._index: HNSW | None = None
        self._dim: int | None = None
        self._n: int = 0

        # metric ("l2", "cosine") to match your HNSW implementation
        self._space: str = "l2"

        # store query-time ef separately (your HNSW also has a default ef)
        self._query_ef: int = 50

        # buffered (id -> vector)
        self._buffer: dict[int, np.ndarray] = {}

    def create_index(self, initial_vectors: np.ndarray, **options: Any) -> None:
        x = np.asarray(initial_vectors, dtype=np.float32, order="C")
        if x.ndim != 2:
            raise ValueError(f"initial_vectors must be 2D (N,D); got shape {x.shape}")
        n, d = x.shape
        if n <= 0 or d <= 0:
            raise ValueError(f"invalid shape (N={n}, D={d})")

        # Options (keep same names as before, but map to python HNSW)
        space = str(options.pop("space", "l2"))  # "l2" or "cosine" (ip not supported unless you add it)
        ef_construction = int(options.pop("ef_construction", 200))  # used during add()
        M = int(options.pop("M", 16))  # maps to HNSW(m=...)
        ef = int(options.pop("ef", 50))  # query-time ef
        _num_threads = options.pop("num_threads", None)  # ignored (pure python)
        _max_elements = options.pop("max_elements", None)  # ignored (pure python)

        if options:
            unknown = ", ".join(sorted(options.keys()))
            raise TypeError(f"Unknown option(s) for create_index: {unknown}")

        if space not in ("l2", "cosine"):
            raise ValueError(
                f"space={space!r} not supported by the pasted HNSW code. "
                f"Use 'l2' or 'cosine' (or extend HNSW to support 'ip')."
            )

        # Build the python HNSW
        # - set default ef to query-time ef
        # - use ef_construction per insert to control construction quality
        idx = HNSW(distance_type=space, m=M, ef=ef, heuristic=True, vectorized=False)

        for i in range(n):
            idx.add(x[i], ef=ef_construction)  # IDs become 0..N-1 by insertion order

        self._index = idx
        self._dim = d
        self._n = n
        self._space = space
        self._query_ef = ef
        self._buffer.clear()

    def insert(self, vec_id: int, vec: np.ndarray) -> None:
        if self._dim is None:
            raise RuntimeError("Index not built. Call create_index() first.")

        v = np.asarray(vec, dtype=np.float32, order="C")
        if v.ndim != 1 or v.shape[0] != self._dim:
            raise ValueError(f"vec must be shape ({self._dim},), got {v.shape}")

        # Optional collision check with static ids 0..N-1:
        # if 0 <= vec_id < self._n:
        #     raise ValueError(f"vec_id {vec_id} collides with static id range [0, {self._n-1}]")

        self._buffer[vec_id] = v

    def _buffer_distance(self, q: np.ndarray, v: np.ndarray) -> float:
        """
        Must match the ordering of the underlying HNSW distance:
          smaller = better.
        """
        if self._space == "l2":
            # NOTE: your HNSW uses np.linalg.norm (not squared)
            return float(np.linalg.norm(v - q))

        if self._space == "cosine":
            # Must match your HNSW's cosine distance implementation.
            # If you fixed HNSW.cosine_distance to return 1 - cos_sim, this matches.
            eps = 1e-12
            qn = float(np.linalg.norm(q)) + eps
            vn = float(np.linalg.norm(v)) + eps
            sim = float(np.dot(q, v)) / (qn * vn)
            return float(1.0 - sim)

        raise RuntimeError(f"Unsupported space: {self._space}")

    def query(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._index is None or self._dim is None:
            raise RuntimeError("Index not built. Call create_index() first.")
        if not (1 <= k <= self._n):
            # keeping same behavior as your original (k bounded by static N)
            raise ValueError(f"k must be in [1, {self._n}], got {k}")

        q = np.asarray(q, dtype=np.float32, order="C")

        # Support single query (D,) or batch queries (Q,D)
        if q.ndim == 1:
            return self._query_one(q, k)
        if q.ndim == 2:
            labels_out = np.empty((q.shape[0], k), dtype=np.int64)
            dists_out = np.empty((q.shape[0], k), dtype=np.float32)
            for i, qi in enumerate(q):
                lab, dist = self._query_one(qi, k)
                labels_out[i] = lab
                dists_out[i] = dist
            return labels_out, dists_out

        raise ValueError(f"q must be 1D (D,) or 2D (Q,D); got shape {q.shape}")

    def _query_one(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if q.shape[0] != self._dim:
            raise ValueError(f"dim mismatch: expected {self._dim}, got {q.shape[0]}")

        # 1) Get top-k from static HNSW
        # search() returns list[(idx, dist)] best->worst (ascending dist)
        hits = self._index.search(q, k=k, ef=self._query_ef)

        labels = np.fromiter((idx for idx, _ in hits), dtype=np.int64, count=k)
        dists = np.fromiter((dist for _, dist in hits), dtype=np.float32, count=k)

        # If no buffer, return immediately
        if not self._buffer:
            return labels, dists

        # 2) Replace with better buffered vectors (if any)
        worst_idx = int(np.argmax(dists))
        worst_dist = float(dists[worst_idx])

        for buf_id, buf_vec in self._buffer.items():
            bd = self._buffer_distance(q, buf_vec)
            if bd < worst_dist:
                labels[worst_idx] = buf_id
                dists[worst_idx] = bd
                worst_idx = int(np.argmax(dists))
                worst_dist = float(dists[worst_idx])

        # 3) Re-sort by distance so outputs are in best->worst order
        order = np.argsort(dists)
        return labels[order], dists[order]