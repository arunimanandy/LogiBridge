#!/usr/bin/env python3
"""Full INT8 post-training quantisation using balanced representative samples."""
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model", default="training/models/fp32_saved_model")
    parser.add_argument("--data", default="training/logiedge_dataset.npz")
    parser.add_argument("--out", default="training/models/model_ptq_int8.tflite")
    args = parser.parse_args()

    data = np.load(args.data)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)

    # Balanced representative dataset: include Normal, Warning, Critical.
    rng = np.random.default_rng(42)
    calib_indices = []

    for cls in [0, 1, 2]:
        idx = np.where(y == cls)[0]
        take = min(len(idx), 80)
        calib_indices.extend(rng.choice(idx, size=take, replace=False).tolist())

    # If fewer than 200 samples, fill randomly from full dataset.
    if len(calib_indices) < 200:
        remaining = np.setdiff1d(np.arange(len(X)), np.array(calib_indices))
        extra_take = min(len(remaining), 200 - len(calib_indices))
        calib_indices.extend(rng.choice(remaining, size=extra_take, replace=False).tolist())

    rng.shuffle(calib_indices)
    calib = X[calib_indices].astype(np.float32)

    print("Calibration samples:", len(calib))
    print("Calibration class counts:", {
        int(cls): int(np.sum(y[calib_indices] == cls)) for cls in [0, 1, 2]
    })

    def representative_dataset():
        for row in calib:
            yield [row.reshape(1, -1).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(tflite_model)

    print(f"Wrote {out} ({out.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    main()
