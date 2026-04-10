import random
import threading
import time

import hnsw_cpp


def rand_vec(dim, rng):
    return [rng.random() for _ in range(dim)]


def test_write_write():
    print("=== write + write test ===")

    dim = 16
    per_thread = 500

    index = hnsw_cpp.HNSW()
    index.init(dim=dim, M=16, ef_construction=100, random_seed=123)

    barrier = threading.Barrier(2)
    errors = []
    errors_lock = threading.Lock()

    def writer(name, start_label, seed):
        rng = random.Random(seed)
        try:
            print(f"[{name}] ready")
            barrier.wait()
            print(f"[{name}] starting inserts")

            for i in range(per_thread):
                label = start_label + i
                index.insert(label, rand_vec(dim, rng))

                if (i + 1) % 100 == 0:
                    print(f"[{name}] inserted {i + 1}/{per_thread}")

            print(f"[{name}] done")
        except Exception as e:
            with errors_lock:
                errors.append((name, str(e)))
            print(f"[{name}] ERROR: {e}")

    t1 = threading.Thread(target=writer, args=("writer-1", 0, 1))
    t2 = threading.Thread(target=writer, args=("writer-2", 1_000_000, 2))

    start = time.time()
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    elapsed = time.time() - start

    if errors:
        print("write + write FAILED")
        for name, err in errors:
            print(f"  {name}: {err}")
        return False

    print(f"write + write PASSED in {elapsed:.3f}s")

    # basic sanity search
    q = rand_vec(dim, random.Random(999))
    result = index.knn_search(q, k=5, ef=50)
    print(f"post-write search returned {len(result)} results")
    print()
    return True


def test_read_write():
    print("=== read + write test ===")

    dim = 16
    initial = 500
    writes = 500

    index = hnsw_cpp.HNSW()
    index.init(dim=dim, M=16, ef_construction=100, random_seed=456)

    preload_rng = random.Random(999)
    for label in range(initial):
        index.insert(label, rand_vec(dim, preload_rng))

    print(f"preloaded {initial} vectors")

    stop_event = threading.Event()
    started_event = threading.Event()
    errors = []
    errors_lock = threading.Lock()
    read_count = 0
    read_count_lock = threading.Lock()

    def writer():
        rng = random.Random(111)
        try:
            print("[writer] starting")
            started_event.set()

            for i in range(writes):
                index.insert(10_000 + i, rand_vec(dim, rng))

                if (i + 1) % 100 == 0:
                    print(f"[writer] inserted {i + 1}/{writes}")

                if i % 50 == 0:
                    time.sleep(0.001)

            print("[writer] done")
        except Exception as e:
            with errors_lock:
                errors.append(("writer", str(e)))
            print(f"[writer] ERROR: {e}")
        finally:
            stop_event.set()

    def reader():
        nonlocal read_count
        rng = random.Random(222)

        try:
            started = started_event.wait(timeout=2.0)
            if not started:
                raise RuntimeError("reader timed out waiting for writer to start")

            print("[reader] starting searches")

            while not stop_event.is_set():
                q = rand_vec(dim, rng)
                result = index.knn_search(q, k=10, ef=50)

                if not result:
                    raise RuntimeError("reader got empty result during concurrent write")

                with read_count_lock:
                    read_count += 1
                    local_count = read_count

                if local_count % 100 == 0:
                    print(f"[reader] completed {local_count} searches")

            for _ in range(20):
                q = rand_vec(dim, rng)
                result = index.knn_search(q, k=10, ef=50)
                if not result:
                    raise RuntimeError("reader got empty result after writer finished")

            print("[reader] done")
        except Exception as e:
            with errors_lock:
                errors.append(("reader", str(e)))
            print(f"[reader] ERROR: {e}")

    t_writer = threading.Thread(target=writer)
    t_reader = threading.Thread(target=reader)

    start = time.time()
    t_writer.start()
    t_reader.start()
    t_writer.join()
    t_reader.join()
    elapsed = time.time() - start

    if errors:
        print("read + write FAILED")
        for name, err in errors:
            print(f"  {name}: {err}")
        return False

    print(f"read + write PASSED in {elapsed:.3f}s")
    print(f"reader completed {read_count} searches")
    print()
    return True


if __name__ == "__main__":
    ok1 = test_write_write()
    ok2 = test_read_write()

    if ok1 and ok2:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")