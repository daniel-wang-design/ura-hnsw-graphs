from pathlib import Path

from experiments.harnesses.concurrent_performance_test import *
from implementations.inserting_index import UpdatingHNSWIndex

folder = Path(__file__).resolve().parent
database = UpdatingHNSWIndex()
run(folder=folder, database=database)