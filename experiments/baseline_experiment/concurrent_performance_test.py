import argparse
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import hnsw_cpp

from utility.read_vectors import load_base_and_query


def parse_args(folder: Path):
    parser = argparse.ArgumentParser(
        description="Build the pybind HNSW index, then stress it with concurrent readers and writers."
    )

    parser.add_argument("--base", required=True, type=str, help="Path to base .fbin file")
    parser.add_argument("--query", required=True, type=str, help="Path to query .fbin file")
    parser.add_argument("--k", required=True, type=int, help="Top-k for knn_search")

    parser.add_argument("--space", type=str, default="l2", choices=["l2", "ip", "cosine"])
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef_construction", type=int, default=200)
    parser.add_argument("--ef", type=int, default=50)
    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--num_readers", type=int, default=100)
    parser.add_argument("--num_writers", type=int, default=100)
    parser.add_argument(
        "--insert_query_idx",
        type=int,
        default=0,
        help="Every writer inserts query[insert_query_idx] on every write.",
    )
    parser.add_argument(
        "--same_query_for_all_readers",
        action="store_true",
        help="If set, every reader always queries query[insert_query_idx]. Otherwise each reader cycles through query vectors.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(folder / "output.log"),
        help="Path to output log file (default: ./output.log next to this script)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(folder / "output.csv"),
        help="Path to CSV output (default: ./output.csv next to this script)",
    )

    return parser.parse_args()


def to_cpp_vector(vec: np.ndarray) -> list[float]:
    return np.asarray(vec, dtype=np.float64, order="C").tolist()


def summarize_latencies(latencies: list[float]) -> tuple[float, float, float]:
    if not latencies:
        return 0.0, 0.0, 0.0
    arr = np.asarray(latencies, dtype=np.float64)
    return float(arr.mean()), float(np.percentile(arr, 95)), float(arr.max())


def build_index(base: np.ndarray, args) -> tuple[hnsw_cpp.HNSW, float]:
    if args.space != "l2":
        raise ValueError("This pybind HNSW implementation only supports l2 distance")

    if base.ndim != 2 or base.shape[0] == 0:
        raise ValueError(f"base must be a non-empty 2D array, got shape={base.shape}")

    index = hnsw_cpp.HNSW()

    t0 = time.perf_counter()
    index.init(
        dim=int(base.shape[1]),
        M=int(args.M),
        ef_construction=int(args.ef_construction),
        random_seed=int(args.random_seed),
    )

    for label, vec in enumerate(base):
        index.insert(int(label), to_cpp_vector(vec))

    t1 = time.perf_counter()
    return index, (t1 - t0)


def run(folder: Path):
    args = parse_args(folder)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    base, query = load_base_and_query(args.base, args.query)
    base = np.asarray(base, dtype=np.float64, order="C")
    query = np.asarray(query, dtype=np.float64, order="C")

    print("Base vectors:", base.shape, "Query vectors:", query.shape)

    if args.k <= 0:
        raise ValueError(f"k must be > 0; got {args.k}")
    if query.ndim != 2 or query.shape[0] == 0:
        raise ValueError(f"query must be a non-empty 2D array, got shape={query.shape}")
    if base.shape[1] != query.shape[1]:
        raise ValueError(f"base dim {base.shape[1]} != query dim {query.shape[1]}")
    if args.k > base.shape[0]:
        raise ValueError(f"k={args.k} cannot exceed initial base size N={base.shape[0]}")
    if not (0 <= args.insert_query_idx < query.shape[0]):
        raise ValueError(
            f"insert_query_idx must be in [0, {query.shape[0] - 1}], got {args.insert_query_idx}"
        )

    index, build_secs = build_index(base, args)

    N, D = base.shape
    Q = query.shape[0]
    insert_vec = np.asarray(query[args.insert_query_idx], dtype=np.float64, order="C").copy()

    start_event = threading.Event()
    failure_event = threading.Event()
    id_lock = threading.Lock()
    exceptions_lock = threading.Lock()

    next_insert_id = N
    exceptions: list[tuple[str, int, str]] = []
    reader_results = [None] * args.num_readers
    writer_results = [None] * args.num_writers

    def alloc_insert_id() -> int:
        nonlocal next_insert_id
        with id_lock:
            out = next_insert_id
            next_insert_id += 1
            return out

    def record_exception(worker_type: str, worker_id: int):
        tb = traceback.format_exc()
        with exceptions_lock:
            exceptions.append((worker_type, worker_id, tb))
        failure_event.set()

    def reader_worker(reader_id: int):
        latencies = []
        completed = 0
        t0 = None
        t1 = None

        try:
            print(f"[reader-{reader_id}] ready")
            start_event.wait()
            print(f"[reader-{reader_id}] starting")

            t0 = time.perf_counter()
            for i in range(Q):
                if failure_event.is_set():
                    break

                if args.same_query_for_all_readers:
                    qv = insert_vec
                else:
                    qv = query[(reader_id + i) % Q]

                tq0 = time.perf_counter()
                result = index.knn_search(to_cpp_vector(qv), int(args.k), int(args.ef))
                tq1 = time.perf_counter()

                if not result:
                    raise RuntimeError("reader got empty result")

                latencies.append(tq1 - tq0)
                completed += 1

            t1 = time.perf_counter()
            print(f"[reader-{reader_id}] done {completed}/{Q}")
        except Exception:
            t1 = time.perf_counter()
            record_exception("reader", reader_id)
            print(f"[reader-{reader_id}] ERROR")

        avg_s, p95_s, max_s = summarize_latencies(latencies)
        wall_s = (t1 - t0) if (t0 is not None and t1 is not None) else 0.0
        reader_results[reader_id] = {
            "worker_type": "reader",
            "worker_id": reader_id,
            "completed_ops": completed,
            "target_ops": Q,
            "wall_seconds": wall_s,
            "avg_latency_seconds": avg_s,
            "p95_latency_seconds": p95_s,
            "max_latency_seconds": max_s,
        }

    def writer_worker(writer_id: int):
        latencies = []
        completed = 0
        t0 = None
        t1 = None

        try:
            print(f"[writer-{writer_id}] ready")
            start_event.wait()
            print(f"[writer-{writer_id}] starting")

            t0 = time.perf_counter()
            for _ in range(Q):
                if failure_event.is_set():
                    break

                label = alloc_insert_id()
                ti0 = time.perf_counter()
                index.insert(int(label), to_cpp_vector(insert_vec))
                ti1 = time.perf_counter()

                latencies.append(ti1 - ti0)
                completed += 1

            t1 = time.perf_counter()
            print(f"[writer-{writer_id}] done {completed}/{Q}")
        except Exception:
            t1 = time.perf_counter()
            record_exception("writer", writer_id)
            print(f"[writer-{writer_id}] ERROR")

        avg_s, p95_s, max_s = summarize_latencies(latencies)
        wall_s = (t1 - t0) if (t0 is not None and t1 is not None) else 0.0
        writer_results[writer_id] = {
            "worker_type": "writer",
            "worker_id": writer_id,
            "completed_ops": completed,
            "target_ops": Q,
            "wall_seconds": wall_s,
            "avg_latency_seconds": avg_s,
            "p95_latency_seconds": p95_s,
            "max_latency_seconds": max_s,
        }

    threads = []
    for reader_id in range(args.num_readers):
        threads.append(threading.Thread(target=reader_worker, args=(reader_id,), name=f"reader-{reader_id}"))
    for writer_id in range(args.num_writers):
        threads.append(threading.Thread(target=writer_worker, args=(writer_id,), name=f"writer-{writer_id}"))

    for t in threads:
        t.start()

    t_concurrent0 = time.perf_counter()
    start_event.set()

    for t in threads:
        t.join()
    t_concurrent1 = time.perf_counter()

    def safe_mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    def safe_max(values: list[float]) -> float:
        return float(np.max(values)) if values else 0.0

    reader_completed = sum(r["completed_ops"] for r in reader_results if r is not None)
    writer_completed = sum(w["completed_ops"] for w in writer_results if w is not None)
    reader_wall_values = [r["wall_seconds"] for r in reader_results if r is not None]
    writer_wall_values = [w["wall_seconds"] for w in writer_results if w is not None]
    total_elapsed_s = t_concurrent1 - t_concurrent0

    log_lines = []
    log_lines.append("=== Concurrent HNSW Performance Test ===")
    log_lines.append(f"base_path:           {args.base}")
    log_lines.append(f"query_path:          {args.query}")
    log_lines.append(f"output:              {out_path}")
    log_lines.append(f"csv:                 {csv_path}")
    log_lines.append("")
    log_lines.append(f"initial_N:           {N}")
    log_lines.append(f"Q (ops per worker):  {Q}")
    log_lines.append(f"D:                   {D}")
    log_lines.append(f"k:                   {args.k}")
    log_lines.append(f"num_readers:         {args.num_readers}")
    log_lines.append(f"num_writers:         {args.num_writers}")
    log_lines.append(f"insert_query_idx:    {args.insert_query_idx}")
    log_lines.append(f"same_query_for_all:  {args.same_query_for_all_readers}")
    log_lines.append("")
    log_lines.append("=== Build ===")
    log_lines.append(f"space:               {args.space}")
    log_lines.append(f"M:                   {args.M}")
    log_lines.append(f"ef_construction:     {args.ef_construction}")
    log_lines.append(f"ef:                  {args.ef}")
    log_lines.append(f"random_seed:         {args.random_seed}")
    log_lines.append(f"build_seconds:       {build_secs:.6f}")
    log_lines.append("")
    log_lines.append("=== Concurrent phase ===")
    log_lines.append(f"elapsed_seconds:     {total_elapsed_s:.6f}")
    log_lines.append(f"reader_ops_done:     {reader_completed}")
    log_lines.append(f"writer_ops_done:     {writer_completed}")
    log_lines.append(f"expected_reader_ops: {args.num_readers * Q}")
    log_lines.append(f"expected_writer_ops: {args.num_writers * Q}")
    log_lines.append(f"exceptions:          {len(exceptions)}")
    log_lines.append("")
    log_lines.append("=== Reader summary ===")
    log_lines.append(f"avg_wall_seconds:    {safe_mean(reader_wall_values):.6f}")
    log_lines.append(f"max_wall_seconds:    {safe_max(reader_wall_values):.6f}")
    log_lines.append("")
    log_lines.append("=== Writer summary ===")
    log_lines.append(f"avg_wall_seconds:    {safe_mean(writer_wall_values):.6f}")
    log_lines.append(f"max_wall_seconds:    {safe_max(writer_wall_values):.6f}")

    if exceptions:
        log_lines.append("")
        log_lines.append("=== Exceptions ===")
        for worker_type, worker_id, tb in exceptions:
            log_lines.append(f"[{worker_type}-{worker_id}]")
            log_lines.append(tb.rstrip())
            log_lines.append("")

    csv_rows = [
        "worker_type,worker_id,completed_ops,target_ops,wall_seconds,avg_latency_seconds,p95_latency_seconds,max_latency_seconds"
    ]
    for result in [r for r in reader_results if r is not None] + [w for w in writer_results if w is not None]:
        csv_rows.append(
            f"{result['worker_type']},{result['worker_id']},{result['completed_ops']},{result['target_ops']},"
            f"{result['wall_seconds']:.6f},{result['avg_latency_seconds']:.6f},"
            f"{result['p95_latency_seconds']:.6f},{result['max_latency_seconds']:.6f}"
        )

    out_path.write_text("\n".join(log_lines), encoding="utf-8")
    csv_path.write_text("\n".join(csv_rows) + "\n", encoding="utf-8")

    print(f"HNSW build: {build_secs:.3f}s")
    print(f"Concurrent phase: {total_elapsed_s:.3f}s")
    print(f"Readers completed: {reader_completed}/{args.num_readers * Q}")
    print(f"Writers completed: {writer_completed}/{args.num_writers * Q}")
    print(f"Exceptions: {len(exceptions)}")
    print(f"Wrote log to: {out_path}")
    print(f"Wrote CSV to: {csv_path}")


def main():
    folder = Path(__file__).resolve().parent
    run(folder)


if __name__ == "__main__":
    main()
