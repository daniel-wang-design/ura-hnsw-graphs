# Test Data

### sequential_insert_baseline
The database vectors are generated using gaussian-like random. The queries are generated with temporal locality. That is, the vectors are close to a single base vector by random gaussian distribution.

### sequential_insert_multiple_hotspots
The database vectors are randomly generated. The query vectors are generated with hotspot locality. There are multiple hotspots represents various popular topics.

# Experiments

### Baseline Experiment

This experiment involved having client A insert new vectors while client B queries top k after each insert. The index is not updated so we expect the accuracy to decrease.

To run

PowerShell:
```
python -m experiments.baseline_experiment.run --base .\testcases\sequential_insert_baseline\base_vectors\base_vectors.fbin --query .\testcases\sequential_insert_baseline\query_vectors\query_vectors.fbin --k 100 --ef 500
```
Bash
```
python -m experiments.baseline_experiment.run_sequential_creation --base ./testcases/sequential_insert_baseline/base_vec
tors/base_vectors.fbin --query ./testcases/sequential_insert_baseline/query_vectors/query_vectors.fbin --k 100 --ef 500
```

This is the result of multiple insert hotspots with temporal locality for each query ie query after insert.

![alt text](image-1.png)

The graph below is with single hotspot inset and temporal locality query ie query after insert.

![alt text](image-3.png)


The graphs below include probabilistic temporal distribution. It takes two parameters, percentage, and partition. Partition is what percentage of the query vectors will be inserted. Percentage is the percentage of how many of the query vectors come from already inserted vectors, with replacement. The rest of the query vectors come from the remaining partition that will never be inserted.

To run:

```
python -m experiments.baseline_experiment.run_probability --base ./testcases/sequential_insert_baseline/base_vectors/base_vectors.fbin  --query ./testcases/sequential_insert_baseline/query_vectors/query_vectors.fbin  --k 100 --ef 500 --partit
ion 90 --probability 90
```

The graph below is single hotspot:

![alt text](image-4.png)

The graph below is for multiple hotspots:

![alt text](image-5.png)

### Buffer Experiment
This experiment adds a basic buffer to the baseline. The index is not changed ever. For each query, simply use the old index to get topk, then replace the furthest vectors with closer vectors from the buffer.

To run
```
python -m experiments.buffer_experiment.run --base ./testcases/sequential_insert_baseline/base_vectors/base_vectors.fbin  --query ./testcases/sequential_insert_baseline/query_vectors/query_vectors.fbin --k 100 --ef 500
```

The graph below is with multiple insert hotspots and each query has temporal locality ie query after insert.

![alt text](image.png)

The graph below is with single hotspot inset and temporal locality query ie query after insert. 

![alt text](image-2.png)

The graphs below are with probabilitistic termporal query insert/query, with probability = 90, and partition = 90.

The graph below is for single hotspot:

![alt text](image-7.png)

The graph below is for multiple hotspots:

![alt text](image-6.png)

### Sequential vs Batch Index Creation

The graph below is for batch index creation

![alt text](image-1.png)
![alt text](image-3.png)

The graph below is for sequential index creation

![alt text](image-8.png)
![alt text](image-9.png)

Below are the results of running recall with multiple hotspots using the new HNSW

![alt text](image-10.png)

The graph below is using the C++ implementation:

![alt text](image-11.png)

## CPP HNSW Locking Results

Using optimized code:
```
HNSW build: 158.783s
Concurrent phase: 440.352s
Readers completed: 1000000/1000000
Writers completed: 1000000/1000000
Exceptions: 0
```

Using no-ops to mimic no concurrency control:
```
double free or corruption (!prev)
Aborted (core dumped)
```

Using hnswlib
```
Concurrent phase: 0.406s
Readers completed: 2427/1000000
Writers completed: 239/1000000
Exceptions: 8
```

Logging the number of time we wait for a lock and time spent waiting for a lock:
```
Build: 263.644s
Concurrent phase: 1010.192s
Readers completed: 1000000/1000000
Writers completed: 1000000/1000000
Exceptions: 0

=== Node lock wait stats ===
Lock waits: 882596
Total wait time: 678.014921 seconds
```

Sanity test to ensure that we are actually inserting vectors:
```
ura-hnsw-graphs/hnsw$ PYTHONPATH=./hnsw:. python test.py
Test passed: inserted vector was found by query.
```