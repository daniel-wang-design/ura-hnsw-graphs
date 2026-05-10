# How to compile to Python from C++
```
g++ -O3 -Wall -shared -std=c++17 -fPIC \
  $(python3-config --includes) \
  -I/usr/include/pybind11 \
  hnsw_per_node_locks_optmized_with_counters.cpp \
  -o hnsw_cpp$(python3-config --extension-suffix)
```

# To run with C++ HNSW

Use WSL with python3 venv
```
python3 -m venv venv3
```
or

```
source venv3/bin/activate
```

Install packages from requirements.txt

Run command:
```
PYTHONPATH=./hnsw:. python -m experiments.baseline_experiment.run_pybind --base ./testcases/sequential_insert_multiple_hotspots/base_vectors/base_vectors.fbin --query ./testcases/sequential_insert_multiple_hotspots/query_vectors/query_vectors.fbin --k 100 --ef 500
```

Running performance tests:
```
PYTHONPATH=./hnsw:. python -m experiments.baseline_experiment.run_cpp_with_lock_count --base ./testcases/sequential_insert_baseline/base_vectors/base_vectors.fbin --query ./testcases/sequential_insert_baseline/query_vectors/query_vectors.fbin --k 100
```
