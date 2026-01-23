import numpy as np
from typing import Tuple

def brute_force_knn_l2(
    base: np.ndarray,
    query: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact brute-force kNN under squared L2 distance.

    Args:
      base:  (N, D) float32
      query: (Q, D) float32
      k:     number of neighbors (1 <= k <= N)

    Returns:
      idx:   (Q, k) int64 indices into base
      dist:  (Q, k) float32 squared L2 distances
    """
    if base.ndim != 2 or query.ndim != 2:
        raise ValueError("base and query must be 2D arrays (N,D) and (Q,D)")
    if base.shape[1] != query.shape[1]:
        raise ValueError(f"dim mismatch: base {base.shape[1]} vs query {query.shape[1]}")
    if not (1 <= k <= base.shape[0]):
        raise ValueError(f"k must be in [1, N], got k={k}, N={base.shape[0]}")

    base = np.asarray(base, dtype=np.float32, order="C")
    query = np.asarray(query, dtype=np.float32, order="C")

    # dists[q, i] = ||query[q] - base[i]||^2
    base_norm = np.sum(base * base, axis=1)              # (N,)
    query_norm = np.sum(query * query, axis=1)           # (Q,)
    dists = query_norm[:, None] + base_norm[None, :] - 2.0 * (query @ base.T)  # (Q, N)

    # Get k smallest per row without full sort
    idx = np.argpartition(dists, kth=k-1, axis=1)[:, :k]  # (Q, k)

    # Sort those k by distance
    row = np.arange(query.shape[0])[:, None]
    dist_k = dists[row, idx]
    order = np.argsort(dist_k, axis=1)

    idx = idx[row, order]
    dist_k = dist_k[row, order].astype(np.float32, copy=False)

    return idx, dist_k
