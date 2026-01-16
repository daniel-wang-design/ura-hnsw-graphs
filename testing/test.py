import time
import numpy as np
import hnswlib

def l2_exact_knn(x, q, k):
    """
    Exact kNN with L2 distance using brute force.
    Returns indices of the k nearest neighbors for each query.
    """
    # compute squared L2 distances: ||q - x||^2 = ||q||^2 + ||x||^2 - 2 q·x
    x_norm = (x * x).sum(axis=1)[None, :]          # (1, n)
    q_norm = (q * q).sum(axis=1)[:, None]          # (nq, 1)
    dists = q_norm + x_norm - 2.0 * (q @ x.T)      # (nq, n)
    return np.argpartition(dists, kth=k-1, axis=1)[:, :k]

def main():
    rng = np.random.default_rng(42)

    # configuration
    dim = 128
    n = 50_000
    nq = 200
    k = 10

    # random float32 vectors
    x = rng.standard_normal((n, dim), dtype=np.float32)
    q = rng.standard_normal((nq, dim), dtype=np.float32)
    ids = np.arange(n)

    # build HNSW index
    index = hnswlib.Index(space="l2", dim=dim)  # "l2", "cosine", or "ip"
    index.init_index(
        max_elements=n,
        ef_construction=200,
        M=16
    )

    t0 = time.perf_counter()
    index.add_items(x, ids)
    build_s = time.perf_counter() - t0

    # higher ef => better recall, slower queries
    index.set_ef(500)

    # use HNSM
    t0 = time.perf_counter()
    labels, distances = index.knn_query(q, k=k)
    query_s = time.perf_counter() - t0

    # calculate the exact kNN
    t0 = time.perf_counter()
    exact = l2_exact_knn(x, q, k)
    exact_s = time.perf_counter() - t0

    # calculate hits
    hits = 0
    for i in range(nq):
        hits += len(set(labels[i]).intersection(set(exact[i])))
    recall = hits / (nq * k)

    print("=== Results ===")
    print(f"Build time:        {build_s:.3f}s  (n={n}, dim={dim})")
    print(f"HNSW query time:   {query_s:.3f}s  (nq={nq}, k={k})  => {query_s/nq*1e3:.3f} ms/query")
    print(f"Exact query time:  {exact_s:.3f}s  (brute force)")
    print(f"Recall for k={k}:        {recall:.3f}")
    print()
    print("First query neighbors (ids):", labels[0])
    print("First query distances:", distances[0])

if __name__ == "__main__":
    main()
