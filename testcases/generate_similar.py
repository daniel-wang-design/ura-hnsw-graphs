#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

# ========= Output filenames (edit these in one place) =========
OUTPUT_VECTORS_FBIN = "query_vectors.fbin"
OUTPUT_META_JSON = "query_vectors_metadata.json"
# =============================================================


def write_fbin(path: Path, x: np.ndarray) -> None:
    x = np.asarray(x, dtype=np.float32, order="C")
    n, d = x.shape
    with path.open("wb") as f:
        np.asarray([n, d], dtype=np.int32).tofile(f)
        x.tofile(f)


def main():
    ap = argparse.ArgumentParser(
        description="Generate N similar vectors by sampling around K anchor vectors."
    )
    ap.add_argument("--out_dir", required=True, help="Output directory (required)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--dim", type=int, default=128, help="Vector dimension")
    ap.add_argument("--n", type=int, default=10_000, help="Total vectors to generate")
    ap.add_argument(
        "--num_anchors",
        type=int,
        default=200,
        help="Number of anchor vectors (clusters). Smaller => more repetition/similarity.",
    )
    ap.add_argument(
        "--noise_sigma",
        type=float,
        default=0.05,
        help="Stddev of noise around anchor vectors. Smaller => vectors more similar.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Create anchor vectors (cluster centers)
    anchors = rng.standard_normal((args.num_anchors, args.dim), dtype=np.float32)

    # Assign each of N vectors to an anchor, then add small noise
    anchor_idx = rng.integers(0, args.num_anchors, size=args.n, dtype=np.int64)
    noise = rng.normal(0.0, args.noise_sigma, size=(args.n, args.dim)).astype(np.float32)
    x = anchors[anchor_idx] + noise

    vec_path = out_dir / OUTPUT_VECTORS_FBIN
    write_fbin(vec_path, x)

    meta = {
        "seed": int(args.seed),
        "dim": int(args.dim),
        "n": int(args.n),
        "dtype": "float32",
        "distribution": {
            "type": "mixture_of_gaussians",
            "num_anchors": int(args.num_anchors),
            "anchors": "standard_normal",
            "noise": f"Normal(0, {args.noise_sigma}^2) per-dimension",
        },
        "files": {"vectors": OUTPUT_VECTORS_FBIN},
        "formats": {"fbin": "int32 n, int32 dim, then n*dim float32 row-major"},
    }
    with (out_dir / OUTPUT_META_JSON).open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {args.n} vectors (dim={args.dim}) to: {vec_path}")
    print(f"Wrote metadata to: {out_dir / OUTPUT_META_JSON}")


if __name__ == "__main__":
    main()
