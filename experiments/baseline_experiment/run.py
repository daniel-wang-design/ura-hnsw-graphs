from experiments.harnesses.basic_sequential_insert_query import *

from implementations.no_updating_index import StaticHNSWIndex

folder = Path(__file__).resolve().parent
database = StaticHNSWIndex()
run(folder=folder, database=database)