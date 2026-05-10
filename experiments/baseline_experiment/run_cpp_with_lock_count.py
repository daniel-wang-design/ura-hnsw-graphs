from pathlib import Path

from experiments.harnesses.concurrent_performance_test import *
import hnsw_cpp

folder = Path(__file__).resolve().parent
database = hnsw_cpp.HNSW()
run(folder=folder, database=database)

wait_count, wait_seconds = database.node_lock_wait_stats()
print("\n=== Node lock wait stats ===")
print(f"Lock waits: {wait_count}")
print(f"Total wait time: {wait_seconds:.6f} seconds")