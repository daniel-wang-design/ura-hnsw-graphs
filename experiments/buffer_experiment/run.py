from experiments.harnesses.basic_sequential_insert_query import *

from implementations.basic_buffer_index import StaticBufferHNSWIndex

folder = Path(__file__).resolve().parent
database = StaticBufferHNSWIndex()
run(folder=folder, database=database)