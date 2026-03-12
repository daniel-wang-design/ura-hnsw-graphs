from experiments.harnesses.basic_sequential_insert_query import *

from implementations.no_updating_index_cpp import PybindHNSWIndex

folder = Path(__file__).resolve().parent
database = PybindHNSWIndex()
run(folder=folder, database=database)