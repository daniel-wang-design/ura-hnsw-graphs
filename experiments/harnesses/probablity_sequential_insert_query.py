import argparse
import time
from pathlib import Path

import numpy as np

from utility.read_vectors import load_base_and_query
from utility.exact_nearest_neighbor import brute_force_knn_l2
from interfaces.basic_database_interface import BasicHNSWIndex


def parse_args(folder):
    parser = argparse.ArgumentParser(
        description="Run brute-force kNN baseline and compare against an HNSW index "
                    "with partitioned insert/query behavior."
    )

    parser.add_argument("--base", required=True, type=str, help="Path to base .fbin file")
    parser.add_argument("--query", required=True, type=str, help="Path to queries .fbin file")

    parser.add_argument("--k", required=True, type=int, help="Top-k neighbors to retrieve")

    # New behavior controls
    parser.add_argument("--partition", required=True, type=int,
                        help="Percentage of query vectors to use for insertion (0-100)")
    parser.add_argument("--probability", required=True, type=int,
                        help="Probability (percentage) that a query comes from already-inserted vectors")

    script_dir = folder
    parser.add_argument("--output", type=str, default=str(script_dir / "output.log"))
    parser.add_argument("--csv", type=str, default=str(script_dir / "output.csv"))

    parser.add_argument("--space", type=str, default="l2", choices=["l2", "ip", "cosine"])
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef_construction", type=int, default=200)
    parser.add_argument("--ef", type=int, default=50)
    parser.add_argument("--num_threads", type=int, default=-1)

    return parser.parse_args()


def recall_at_k(truth_ids: np.ndarray, approx_ids: np.ndarray, k: int) -> float:
    truth_set = set(map(int, truth_ids[:k].tolist()))
    approx_set = set(map(int, approx_ids[:k].tolist()))
    return len(truth_set & approx_set) / float(k)


def run(folder, database):
    args = parse_args(folder)

    if not (0 < args.partition <= 100):
        raise ValueError("--partition must be in (0, 100]")
    if not (0 <= args.probability <= 100):
        raise ValueError("--probability must be in [0, 100]")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Load vectors
    base, query = load_base_and_query(args.base, args.query)
    base = np.asarray(base, dtype=np.float32, order="C")
    query = np.asarray(query, dtype=np.float32, order="C")

    N, D = base.shape
    Q = query.shape[0]
    k = args.k

    if k <= 0:
        raise ValueError("k must be > 0")
    if k > N:
        raise ValueError(f"k={k} cannot exceed base size N={N}")

    # -------------------------------------------------------------------------
    # Partition queries (by order)
    # -------------------------------------------------------------------------
    num_partition = int((args.partition / 100.0) * Q)
    if num_partition <= 0:
        raise ValueError("Partition size is zero; nothing to insert.")

    partition_queries = query[:num_partition]
    never_inserted_queries = query[num_partition:]

    if len(never_inserted_queries) == 0 and args.probability < 100:
        raise ValueError("No never-inserted queries available, but probability < 100")

    # -------------------------------------------------------------------------
    # Build HNSW index
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
        max_elements=N + num_partition,
    )
    build_secs = time.perf_counter() - t_build0

    growing_base = base
    curr_N = N

    rng = np.random.default_rng()

    log_lines = []
    csv_rows = ["insert_i,recall"]

    log_lines.extend([
        "=== kNN Eval with Partitioned Inserts ===",
        f"base_path:          {args.base}",
        f"query_path:         {args.query}",
        f"partition_percent:  {args.partition}",
        f"probability_percent:{args.probability}",
        f"initial_N:          {N}",
        f"partition_size:     {num_partition}",
        f"never_inserted:     {len(never_inserted_queries)}",
        "",
        "=== HNSW Build ===",
        f"space:             {args.space}",
        f"M:                 {args.M}",
        f"ef_construction:   {args.ef_construction}",
        f"ef (query):        {args.ef}",
        f"build_seconds:     {build_secs:.6f}",
        "",
        "=== Insert → Query Loop ===",
    ])

    total_brute_s = 0.0
    total_hnsw_s = 0.0
    recalls = []

    # -------------------------------------------------------------------------
    # Main loop: insert one, query one
    # -------------------------------------------------------------------------
    for i in range(num_partition):
        insert_vec = partition_queries[i]

        # Insert into brute-force base
        growing_base = np.vstack([growing_base, insert_vec.reshape(1, -1)]).astype(
            np.float32, order="C"
        )
        curr_N += 1

        # Insert into HNSW
        index.insert(N + i, insert_vec)

        # Choose query vector
        use_inserted = rng.random() < (args.probability / 100.0)

        if use_inserted:
            # Sample from already-inserted partition vectors
            qv = partition_queries[rng.integers(0, i + 1)]
            query_source = "inserted"
        else:
            # Sample from never-inserted vectors
            qv = never_inserted_queries[rng.integers(0, len(never_inserted_queries))]
            query_source = "never_inserted"

        # Brute force
        tb0 = time.perf_counter()
        truth_idx, _ = brute_force_knn_l2(growing_base, qv.reshape(1, -1), k)
        tb1 = time.perf_counter()

        # HNSW
        tq0 = time.perf_counter()
        approx_idx, _ = index.query(qv, k)
        tq1 = time.perf_counter()

        truth_ids = np.asarray(truth_idx).reshape(-1)
        approx_ids = np.asarray(approx_idx).reshape(-1)
        rec = recall_at_k(truth_ids, approx_ids, k)

        brute_s = tb1 - tb0
        hnsw_s = tq1 - tq0

        total_brute_s += brute_s
        total_hnsw_s += hnsw_s
        recalls.append(rec)

        csv_rows.append(f"{i},{rec:.6f}")
        log_lines.append(
            f"[insert {i:>6}] N={curr_N:<8} "
            f"query_src={query_source:<14} "
            f"brute_s={brute_s:.6f} hnsw_s={hnsw_s:.6f} recall@{k}={rec:.6f}"
        )

    mean_recall = float(np.mean(recalls)) if recalls else 0.0

    log_lines.extend([
        "",
        "=== Summary ===",
        f"total_inserts:            {num_partition}",
        f"total_brute_seconds:     {total_brute_s:.6f}",
        f"total_hnsw_query_seconds:{total_hnsw_s:.6f}",
        f"mean_recall@{k}:         {mean_recall:.6f}",
    ])

    out_path.write_text("\n".join(log_lines), encoding="utf-8")
    csv_path.write_text("\n".join(csv_rows) + "\n", encoding="utf-8")

    print(f"HNSW build: {build_secs:.3f}s")
    print(f"mean Recall@{k}: {mean_recall:.6f}")
    print(f"Wrote log to: {out_path}")
    print(f"Wrote CSV to: {csv_path}")
