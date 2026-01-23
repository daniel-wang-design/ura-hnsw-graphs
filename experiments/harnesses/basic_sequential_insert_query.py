import argparse
import time
from pathlib import Path

import numpy as np

from utility.read_vectors import load_base_and_query
from utility.exact_nearest_neighbor import brute_force_knn_l2

from interfaces.basic_database_interface import BasicHNSWIndex

def parse_args(folder):
    parser = argparse.ArgumentParser(
        description="Run brute-force kNN baseline and compare against an HNSW index."
    )

    parser.add_argument("--base", required=True, type=str, help="Path to base .fbin file")
    parser.add_argument("--query", required=True, type=str, help="Path to queries .fbin file")

    # required k
    parser.add_argument(
        "--k",
        required=True,
        type=int,
        help="Top-k neighbors to retrieve",
    )

    # output file
    script_dir = folder
    parser.add_argument(
        "--output",
        type=str,
        default=str(script_dir / "output.log"),
        help="Path to output log file (default: ./output.log next to this script)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(script_dir / "output.csv"),
        help="Path to CSV output (default: ./output.csv next to this script)",
    )

    # optional HNSW options (forwarded into create_index)
    parser.add_argument("--space", type=str, default="l2", choices=["l2", "ip", "cosine"])
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef_construction", type=int, default=200)
    parser.add_argument("--ef", type=int, default=50)
    parser.add_argument("--num_threads", type=int, default=-1)

    return parser.parse_args()


def recall_at_k(truth_idx: np.ndarray, approx_idx: np.ndarray, k: int) -> float:
    """
    truth_idx:  (Q, k)
    approx_idx: (Q, k)
    returns mean recall@k = |intersection|/k averaged over queries
    """
    if truth_idx.ndim != 2 or approx_idx.ndim != 2:
        raise ValueError(f"Expected 2D arrays (Q,k). Got {truth_idx.shape=} {approx_idx.shape=}")
    if truth_idx.shape != approx_idx.shape:
        raise ValueError(f"Shape mismatch: {truth_idx.shape} vs {approx_idx.shape}")
    if truth_idx.shape[1] != k:
        raise ValueError(f"Expected second dim == k ({k}); got {truth_idx.shape[1]}")

    hits = 0
    Q = truth_idx.shape[0]
    for i in range(Q):
        # Convert to python sets for intersection size
        hits += len(set(map(int, truth_idx[i])) & set(map(int, approx_idx[i])))
    return hits / (Q * k)


def recall_at_k(truth_ids: np.ndarray, approx_ids: np.ndarray, k: int) -> float:
    truth_set = set(map(int, truth_ids[:k].tolist()))
    approx_set = set(map(int, approx_ids[:k].tolist()))
    return len(truth_set & approx_set) / float(k)


def run(folder, database):
    args = parse_args(folder)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Load vectors
    base, query = load_base_and_query(args.base, args.query)
    base = np.asarray(base, dtype=np.float32, order="C")
    query = np.asarray(query, dtype=np.float32, order="C")

    print("Base vectors:", base.shape, "Query vectors:", query.shape)

    k = int(args.k)
    if k <= 0:
        raise ValueError(f"k must be > 0; got {k}")

    N, D = base.shape
    Q = query.shape[0]

    # -------------------------------------------------------------------------
    # 1) Build HNSW index using base only
    # -------------------------------------------------------------------------
    index: BasicHNSWIndex = database

    t_build0 = time.perf_counter()
    index.create_index(
        base,
        space=args.space,
        ef_construction=args.ef_construction,
        M=args.M,
        ef=args.ef,
        num_threads=args.num_threads,
        max_elements=N + Q,  # allow future inserts
    )
    t_build1 = time.perf_counter()
    build_secs = t_build1 - t_build0

    # We'll maintain a growing base for brute force evaluation.
    # Start with the original base.
    growing_base = base
    curr_N = N  # current number of vectors in growing_base

    if k > curr_N:
        raise ValueError(f"k={k} cannot exceed current base size N={curr_N}")

    log_lines = []
    log_lines.append("=== kNN Eval (Brute vs HNSW) ===")
    log_lines.append(f"base_path:  {args.base}")
    log_lines.append(f"query_path: {args.query}")
    log_lines.append(f"output:     {str(out_path)}")
    log_lines.append("")
    log_lines.append(f"initial_N: {N}, Q: {Q}, D: {D}, k: {k}")
    log_lines.append("")
    log_lines.append("=== HNSW Build ===")
    log_lines.append(f"space:           {args.space}")
    log_lines.append(f"M:               {args.M}")
    log_lines.append(f"ef_construction: {args.ef_construction}")
    log_lines.append(f"ef (query):      {args.ef}")
    log_lines.append(f"num_threads:     {args.num_threads}")
    log_lines.append(f"max_elements:    {N + Q}")
    log_lines.append(f"build_seconds:   {build_secs:.6f}")
    log_lines.append("")

    # -------------------------------------------------------------------------
    # 2) Pre-insert eval using the first query vector (query[0])
    # -------------------------------------------------------------------------
    def eval_one(qv: np.ndarray, eval_tag: str) -> tuple[float, float, float]:
        """
        Returns: (recall, brute_secs, hnsw_secs)
        """
        if k > growing_base.shape[0]:
            raise ValueError(
                f"[{eval_tag}] k={k} cannot exceed current base size N={growing_base.shape[0]}"
            )

        tb0 = time.perf_counter()
        truth_idx, _ = brute_force_knn_l2(growing_base, qv.reshape(1, -1), k)  # (1,k)
        tb1 = time.perf_counter()

        tq0 = time.perf_counter()
        approx_idx, _ = index.query(qv, k)  # (k,)
        tq1 = time.perf_counter()

        truth_ids = np.asarray(truth_idx).reshape(-1)         # (k,)
        approx_ids = np.asarray(approx_idx).reshape(-1)       # (k,)
        rec = recall_at_k(truth_ids, approx_ids, k)
        return rec, (tb1 - tb0), (tq1 - tq0)

    if Q <= 0:
        raise ValueError("Query set is empty")

    rec0_pre, brute0_pre_s, hnsw0_pre_s = eval_one(query[0], "pre_insert_q0")

    log_lines.append("=== Evaluate query[0] before any insertion ===")
    log_lines.append(f"current_N:            {curr_N}")
    log_lines.append(f"q_index:              0")
    log_lines.append(f"brute_seconds:        {brute0_pre_s:.6f}")
    log_lines.append(f"hnsw_query_seconds:   {hnsw0_pre_s:.6f}")
    log_lines.append(f"recall@{k}:           {rec0_pre:.6f}")
    log_lines.append("")

    # -------------------------------------------------------------------------
    # 3) For each query vector i (including i=0):
    #      - add it to growing_base
    #      - insert it into index with new ID
    #      - eval brute vs index and log accuracy
    # -------------------------------------------------------------------------
    log_lines.append("=== Iterative step: Insert each query[i], then evaluate ===")

    total_brute_s = 0.0
    total_hnsw_s = 0.0
    recalls_post = []
    csv_rows = []
    csv_rows.append("insert_i,recall")   # header

    t_loop0 = time.perf_counter()
    for i in range(Q):
        qv = query[i]

        # "add it to base" (for brute force ground truth)
        # Keep C-order float32 for distance compute speed.
        growing_base = np.vstack([growing_base, qv.reshape(1, -1)]).astype(np.float32, order="C")
        curr_N += 1

        # "add it to index" with unique id after original base
        index.insert(N + i, qv)

        # Evaluate after insertion
        rec_i, brute_i_s, hnsw_i_s = eval_one(qv, f"post_insert_q{i}")
        total_brute_s += brute_i_s
        total_hnsw_s += hnsw_i_s
        recalls_post.append(rec_i)
        csv_rows.append(f"{i},{rec_i:.6f}")

        log_lines.append(
            f"[q{i:>6}] after_insert: current_N={curr_N:<8} "
            f"brute_s={brute_i_s:.6f} hnsw_s={hnsw_i_s:.6f} recall@{k}={rec_i:.6f}"
        )

    t_loop1 = time.perf_counter()
    wall_loop_s = t_loop1 - t_loop0

    mean_recall_post = float(np.mean(recalls_post)) if recalls_post else 0.0
    brute_qps_post = Q / total_brute_s if total_brute_s > 0 else float("inf")
    hnsw_qps_post = Q / total_hnsw_s if total_hnsw_s > 0 else float("inf")

    log_lines.append("")
    log_lines.append("=== Summary (post-insert evals) ===")
    log_lines.append(f"total_brute_seconds:      {total_brute_s:.6f}")
    log_lines.append(f"total_hnsw_query_seconds: {total_hnsw_s:.6f}")
    log_lines.append(f"wall_seconds(loop):       {wall_loop_s:.6f}")
    log_lines.append(f"brute_QPS(sum-based):     {brute_qps_post:.2f}")
    log_lines.append(f"hnsw_QPS(sum-based):      {hnsw_qps_post:.2f}")
    log_lines.append(f"mean_recall@{k}:          {mean_recall_post:.6f}")
    log_lines.append("")

    out_path.write_text("\n".join(log_lines), encoding="utf-8")
    csv_path.write_text("\n".join(csv_rows) + "\n", encoding="utf-8")

    print(f"HNSW build: {build_secs:.3f}s")
    print(f"(q0 pre-insert) Recall@{k}: {rec0_pre:.6f}")
    print(f"mean post-insert Recall@{k}: {mean_recall_post:.6f}")
    print(f"Wrote log to: {out_path}")
    print(f"Wrote CSV to: {csv_path}")