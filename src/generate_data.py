"""
Generate synthetic network-flow style data for demo training.

Distributions are inspired by classic IDS benchmarks (e.g. KDD-style features)
but fully synthetic — no third-party dataset redistribution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from features import FEATURE_COLUMNS, LABEL_MAP
except ImportError:
    from src.features import FEATURE_COLUMNS, LABEL_MAP


def _benign(n: int, rng: np.random.Generator) -> np.ndarray:
    X = np.zeros((n, len(FEATURE_COLUMNS)))
    X[:, 0] = rng.exponential(2.0, n)  # duration
    X[:, 1] = rng.choice([0, 1, 2], n, p=[0.7, 0.25, 0.05])
    X[:, 2] = rng.lognormal(6, 1.2, n)  # src_bytes
    X[:, 3] = rng.lognormal(6.5, 1.3, n)  # dst_bytes
    X[:, 16] = rng.poisson(8, n)  # count
    X[:, 17] = rng.poisson(6, n)  # srv_count
    X[:, 22] = rng.uniform(0.7, 1.0, n)  # same_srv_rate
    X[:, 24] = rng.integers(1, 50, n)  # dst_host_count
    X[:, 25] = rng.integers(1, 40, n)
    X[:, 26] = rng.uniform(0.6, 1.0, n)
    X[:, 8] = rng.choice([0, 1], n, p=[0.3, 0.7])  # logged_in
    return X


def _probe(n: int, rng: np.random.Generator) -> np.ndarray:
    X = _benign(n, rng)
    X[:, 0] = rng.exponential(0.3, n)
    X[:, 2] = rng.integers(0, 50, n)
    X[:, 3] = rng.integers(0, 50, n)
    X[:, 16] = rng.integers(40, 200, n)  # high connection count
    X[:, 17] = rng.integers(30, 180, n)
    X[:, 22] = rng.uniform(0.0, 0.3, n)  # many different services
    X[:, 23] = rng.uniform(0.5, 1.0, n)
    X[:, 24] = rng.integers(80, 255, n)
    X[:, 27] = rng.uniform(0.4, 1.0, n)
    return X


def _dos(n: int, rng: np.random.Generator) -> np.ndarray:
    X = _benign(n, rng)
    X[:, 0] = rng.exponential(0.1, n)
    X[:, 2] = rng.lognormal(10, 0.8, n)  # flood volume
    X[:, 3] = rng.integers(0, 100, n)
    X[:, 16] = rng.integers(100, 500, n)
    X[:, 17] = rng.integers(80, 400, n)
    X[:, 18] = rng.uniform(0.5, 1.0, n)  # serror_rate
    X[:, 19] = rng.uniform(0.4, 1.0, n)
    X[:, 28] = rng.uniform(0.4, 1.0, n)
    return X


def _r2l(n: int, rng: np.random.Generator) -> np.ndarray:
    X = _benign(n, rng)
    X[:, 7] = rng.integers(1, 8, n)  # failed logins
    X[:, 8] = 0
    X[:, 2] = rng.integers(0, 200, n)
    X[:, 3] = rng.integers(0, 200, n)
    X[:, 5] = rng.integers(0, 3, n)
    return X


def _u2r(n: int, rng: np.random.Generator) -> np.ndarray:
    X = _benign(n, rng)
    X[:, 6] = rng.integers(1, 10, n)  # hot indicators
    X[:, 9] = rng.integers(1, 5, n)  # compromised
    X[:, 10] = rng.choice([0, 1], n, p=[0.4, 0.6])  # root_shell
    X[:, 11] = rng.choice([0, 1], n, p=[0.5, 0.5])
    X[:, 12] = rng.integers(0, 5, n)
    X[:, 13] = rng.integers(0, 5, n)
    X[:, 14] = rng.integers(0, 3, n)
    return X


GENERATORS = {
    0: _benign,
    1: _probe,
    2: _dos,
    3: _r2l,
    4: _u2r,
}


def generate(n_samples: int = 12000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Class mix: mostly benign, minority attacks
    weights = np.array([0.72, 0.10, 0.10, 0.05, 0.03])
    counts = (weights * n_samples).astype(int)
    counts[-1] = n_samples - counts[:-1].sum()

    blocks = []
    labels = []
    for cls, n in enumerate(counts):
        blocks.append(GENERATORS[cls](n, rng))
        labels.append(np.full(n, cls, dtype=int))

    X = np.vstack(blocks)
    y = np.concatenate(labels)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]

    df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    df["label"] = y
    df["label_name"] = df["label"].map(LABEL_MAP)
    df["is_attack"] = (df["label"] > 0).astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Aegis training data")
    parser.add_argument("--n", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/flows.csv")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.n, args.seed)
    df.to_csv(out, index=False)
    print(f"[+] Wrote {len(df)} rows → {out}")
    print(df["label_name"].value_counts().to_string())


if __name__ == "__main__":
    main()
