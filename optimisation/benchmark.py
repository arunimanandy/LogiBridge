#!/usr/bin/env python3
"""Benchmark FP32/INT8/pruned variants: latency, p95, size, accuracy, energy."""
import argparse, csv, os, time
from pathlib import Path
import numpy as np
import psutil
from sklearn.metrics import accuracy_score, recall_score

try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except Exception:
    from tflite_runtime.interpreter import Interpreter

def run_interpreter(model_path, X):
    itp = Interpreter(model_path=str(model_path))
    itp.allocate_tensors()
    inp = itp.get_input_details()[0]
    outd = itp.get_output_details()[0]
    preds, lats = [], []
    for i, x in enumerate(X[:210]):
        qx = x.astype(np.float32)
        if inp["dtype"] == np.int8:
            scale, zero = inp["quantization"]
            qx = np.clip(qx / scale + zero, -128, 127).astype(np.int8)
        t0 = time.perf_counter()
        itp.set_tensor(inp["index"], qx.reshape(1, -1))
        itp.invoke()
        y = itp.get_tensor(outd["index"])[0]
        t1 = time.perf_counter()
        if i >= 10:
            lats.append((t1 - t0) * 1000)
            preds.append(int(np.argmax(y)))
    return np.array(preds), np.array(lats)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training/logiedge_dataset.npz")
    parser.add_argument("--models", default="training/models")
    parser.add_argument("--out", default="optimisation/results/benchmark_results.csv")
    parser.add_argument("--tdp-watts", type=float, default=15.0)
    args = parser.parse_args()

    data = np.load(args.data)
    X, y = data["X"].astype(np.float32), data["y"].astype(np.int64)
    y_eval = y[10:210]
    model_paths = {
        "M1_FP32": Path(args.models) / "model_fp32.tflite",
        "M2_PTQ_INT8": Path(args.models) / "model_ptq_int8.tflite",
        "M3_PRUNED_INT8": Path(args.models) / "model_pruned_int8.tflite",
    }
    rows = []
    for name, path in model_paths.items():
        if not path.exists():
            print(f"Skipping missing model {path}")
            continue
        cpu_before = psutil.cpu_percent(interval=0.2)
        pred, lats = run_interpreter(path, X)
        cpu_after = psutil.cpu_percent(interval=0.2)
        mean_ms = float(lats.mean())
        p95_ms = float(np.percentile(lats, 95))
        power = args.tdp_watts * max(cpu_before, cpu_after, 1.0) / 100.0
        energy_mj = power * (mean_ms / 1000.0) * 1000.0
        rows.append({
            "variant": name,
            "mean_latency_ms": f"{mean_ms:.3f}",
            "p95_latency_ms": f"{p95_ms:.3f}",
            "model_size_kb": f"{path.stat().st_size/1024:.1f}",
            "accuracy_percent": f"{accuracy_score(y_eval[:len(pred)], pred)*100:.2f}",
            "class2_recall_percent": f"{recall_score(y_eval[:len(pred)], pred, labels=[2], average=None, zero_division=0)[0]*100:.2f}",
            "energy_mj": f"{energy_mj:.3f}"
        })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["variant"])
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
