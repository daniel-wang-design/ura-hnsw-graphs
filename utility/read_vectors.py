from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def read_fbin(path: str | Path) -> np.ndarray:
    """
    Read vectors from a .fbin file with layout:
      int32 n, int32 dim, then n*dim float32 row-major.

    Returns:
      np.ndarray of shape (n, dim), dtype float32
    """
    path = Path(path)

    # Fast file-size sanity before reading lots of data
    file_size = path.stat().st_size
    if file_size < 8:
        raise ValueError(f"{path}: file too small ({file_size} bytes) to contain fbin header")

    with path.open("rb") as f:
        hdr = np.fromfile(f, dtype=np.int32, count=2)
        if hdr.size != 2:
            raise ValueError(f"{path}: failed to read 2 int32 header values")

        n = int(hdr[0])
        d = int(hdr[1])

        # Header sanity checks (tune thresholds if you truly expect bigger)
        if n <= 0 or d <= 0:
            raise ValueError(f"{path}: invalid header n={n}, d={d}")
        if d > 1_000_000 or n > 1_000_000_000:
            raise ValueError(f"{path}: suspicious header n={n}, d={d} (wrong file/format/endian?)")

        # Exact size check for this format: 8 header bytes + n*d*4 payload bytes
        expected_size = 8 + (n * d * 4)
        if file_size != expected_size:
            raise ValueError(
                f"{path}: size mismatch. header says n={n}, d={d} => expected {expected_size} bytes, got {file_size} bytes"
            )

        # Now it's safe to read
        x = np.fromfile(f, dtype=np.float32, count=n * d)
        if x.size != n * d:
            raise ValueError(f"{path}: truncated payload (expected {n*d} floats, got {x.size})")

    return x.reshape(n, d)


def load_base_and_query(base_fbin: str | Path, query_fbin: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    base = read_fbin(base_fbin)
    query = read_fbin(query_fbin)

    if base.shape[1] != query.shape[1]:
        raise ValueError(f"Dim mismatch: base dim={base.shape[1]} vs query dim={query.shape[1]}")

    base = np.asarray(base, dtype=np.float32, order="C")
    query = np.asarray(query, dtype=np.float32, order="C")
    return base, query
