#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def write_fbin(path: Path, x: np.ndarray) -> None:
    x = np.asarray(x, dtype=np.float32, order="C")
    n, d = x.shape
    with path.open("wb") as f:
        np.asarray([n, d], dtype=np.int32).tofile(f)
        x.tofile(f)


def main():
    ap = argparse.ArgumentParser(description="Generate N base vectors for ANN tests.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory (required)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--dim", type=int, default=128, help="Vector dimension")
    ap.add_argument("--n", type=int, default=50_000, help="Number of base vectors (N)")
    ap.add_argument("--format", type=str, default="fbin", choices=["fbin", "npy"], help="Output format")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # same distribution & dtype as your prior test
    x = rng.standard_normal((args.n, args.dim), dtype=np.float32)
    
    # file names
    METADATA = 'base_vectors_metadata.json'
    OUTPUT = 'base_vectors'
    
    # save vectors
    if args.format == "fbin":
        vec_path = out_dir / f"{OUTPUT}.fbin"
        write_fbin(vec_path, x)
    else:
        vec_path = out_dir / f"{OUTPUT}.npy"
        np.save(vec_path, x)

    # minimal metadata
    meta = {
        "seed": args.seed,
        "dim": args.dim,
        "n_base": args.n,
        "dtype": "float32",
        "distribution": "standard_normal",
        "file": vec_path.name,
        "formats": {
            "fbin": "int32 n, int32 dim, n*dim float32 row-major"
        }
    }
    with (out_dir / METADATA).open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {args.n} base vectors (dim={args.dim}) to: {vec_path}")
    print(f"Wrote metadata to: {out_dir / METADATA}")


if __name__ == "__main__":
    main()
