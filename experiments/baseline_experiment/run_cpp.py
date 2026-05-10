from pathlib import Path

from experiments.harnesses.concurrent_performance_test import *
import hnsw_cpp

folder = Path(__file__).resolve().parent
database = hnsw_cpp.HNSW()
run(folder=folder, database=database)