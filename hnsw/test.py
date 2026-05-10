import hnsw_cpp


def main():
    index = hnsw_cpp.HNSW()

    vector = [1.0, 2.0, 3.0]
    label = 123

    index.init(dim=3, M=4, ef_construction=10, random_seed=42)
    index.insert(label, vector)

    results = index.knn_search(vector, k=1, ef=10)

    assert len(results) == 1, "Expected one query result"

    returned_label, distance = results[0]

    assert returned_label == label, (
        f"Expected label {label}, got {returned_label}"
    )

    assert distance == 0.0, (
        f"Expected distance 0.0 for identical vector, got {distance}"
    )

    print("Test passed: inserted vector was found by query.")


if __name__ == "__main__":
    main()