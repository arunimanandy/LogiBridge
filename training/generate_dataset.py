#!/usr/bin/env python3
"""Generate labelled feature dataset from the simulator model."""
import argparse, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from data_pipeline.preprocessing import simulate_feature_matrix

DURATIONS = {0: ("none", 20), 1: ("temp_drift", 15), 2: ("combined", 15)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="training/logiedge_dataset.npz")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    Xs, ys = [], []
    for label, (mode, minutes) in DURATIONS.items():
        X = simulate_feature_matrix(minutes=minutes, anomaly=mode, seed=args.seed + label)
        y = np.full(len(X), label, dtype=np.int64)
        Xs.append(X); ys.append(y)
    X = np.vstack(Xs).astype(np.float32)
    y = np.concatenate(ys)

    stats = {"mean": Xs[0].mean(axis=0), "std": Xs[0].std(axis=0) + 1e-6}
    Xn = (X - stats["mean"]) / stats["std"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=Xn, y=y, raw_X=X, mean=stats["mean"], std=stats["std"])
    np.save(ROOT / "data_pipeline" / "training_stats.npy", stats)
    print(f"Saved {len(X)} windows to {out}")
    print("Class counts:", {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))})

if __name__ == "__main__":
    main()
