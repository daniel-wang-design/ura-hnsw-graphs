from experiments.harnesses.basic_sequential_insert_query import *

from implementations.no_updating_index_sequential_creation import StaticHNSWIndexSequentialCreation

folder = Path(__file__).resolve().parent
database = StaticHNSWIndexSequentialCreation()
run(folder=folder, database=database)