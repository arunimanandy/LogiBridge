#!/usr/bin/env python3
"""Full INT8 post-training quantisation using representative samples."""
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

    X = np.load(args.data)["X"].astype(np.float32)
    calib = X[:max(200, min(len(X), 200))]
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
    print(f"Wrote {out} ({out.stat().st_size/1024:.1f} KB)")

if __name__ == "__main__":
    main()
