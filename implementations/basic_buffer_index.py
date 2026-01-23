from __future__ import annotations

from typing import Any
import numpy as np
import hnswlib

from interfaces.basic_database_interface import BasicHNSWIndex


class StaticBufferHNSWIndex(BasicHNSWIndex):
    """
    Static HNSW index + buffered inserts:
      - create_index() builds from initial_vectors (static)
      - insert() buffers vectors (no index update)
      - query() gets top-k from HNSW, then replaces with better buffered vectors if any
    """

    def __init__(self) -> None:
        self._index: hnswlib.Index | None = None
        self._dim: int | None = None
        self._n: int = 0

        # metric used by hnswlib ("l2", "ip", "cosine")
        self._space: str = "l2"

        # buffered (id -> vector). dict is simplest.
        self._buffer: dict[int, np.ndarray] = {}

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
        max_elements = int(options.pop("max_elements", n))
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
        self._space = space
        self._buffer.clear()

    def insert(self, vec_id: int, vec: np.ndarray) -> None:
        if self._dim is None:
            raise RuntimeError("Index not built. Call create_index() first.")

        v = np.asarray(vec, dtype=np.float32, order="C")
        if v.ndim != 1 or v.shape[0] != self._dim:
            raise ValueError(f"vec must be shape ({self._dim},), got {v.shape}")

        # If you want to forbid collisions with static ids 0..N-1, uncomment:
        # if 0 <= vec_id < self._n:
        #     raise ValueError(f"vec_id {vec_id} collides with static id range [0, {self._n-1}]")

        self._buffer[vec_id] = v

    def _buffer_distance(self, q: np.ndarray, v: np.ndarray) -> float:
        """
        Compute a "distance-like" score consistent with hnswlib's ordering:
          smaller = better.
        """
        if self._space == "l2":
            diff = v - q
            return float(np.dot(diff, diff))  # squared L2

        if self._space == "cosine":
            eps = 1e-12
            qn = float(np.linalg.norm(q)) + eps
            vn = float(np.linalg.norm(v)) + eps
            sim = float(np.dot(q, v)) / (qn * vn)
            return float(1.0 - sim)

        if self._space == "ip":
            # hnswlib ranks by increasing "distance". For IP, it treats higher dot as better,
            # so a common compatible score is -dot(q, v).
            return float(-np.dot(q, v))

        raise RuntimeError(f"Unsupported space: {self._space}")

    def query(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._index is None or self._dim is None:
            raise RuntimeError("Index not built. Call create_index() first.")
        if not (1 <= k <= self._n):
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
        labels, dists = self._index.knn_query(q.reshape(1, self._dim), k=k)
        labels = labels[0].astype(np.int64, copy=False)
        dists = dists[0].astype(np.float32, copy=False)

        # If no buffer, return immediately
        if not self._buffer:
            return labels, dists

        # 2) Replace with better buffered vectors (if any)
        # We'll maintain the current "worst" (largest distance) among the top-k.
        # When we find a buffer point better than the worst, we replace that worst slot.
        worst_idx = int(np.argmax(dists))
        worst_dist = float(dists[worst_idx])

        for buf_id, buf_vec in self._buffer.items():
            bd = self._buffer_distance(q, buf_vec)
            if bd < worst_dist:
                labels[worst_idx] = buf_id
                dists[worst_idx] = bd

                # update worst
                worst_idx = int(np.argmax(dists))
                worst_dist = float(dists[worst_idx])

        # 3) Re-sort by distance so outputs are in best->worst order
        order = np.argsort(dists)
        return labels[order], dists[order]
