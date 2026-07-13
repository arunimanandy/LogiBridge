#!/usr/bin/env python3
"""Structured pruning followed by INT8 PTQ."""
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from sklearn.model_selection import train_test_split
from train_model import build_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training/logiedge_dataset.npz")
    parser.add_argument("--out", default="training/models/model_pruned_int8.tflite")
    args = parser.parse_args()
    data = np.load(args.data)
    X, y = data["X"].astype(np.float32), data["y"].astype(np.int64)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    base = build_model(X.shape[1])
    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0, final_sparsity=0.35, begin_step=0, end_step=1000)
    }
    model = tfmot.sparsity.keras.prune_low_magnitude(base, **pruning_params)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=40, callbacks=callbacks, verbose=2)
    stripped = tfmot.sparsity.keras.strip_pruning(model)

    tmp = Path("training/models/pruned_saved_model")
    stripped.export(tmp)
    calib = X[:max(200, min(len(X), 200))]
    def representative_dataset():
        for row in calib:
            yield [row.reshape(1, -1).astype(np.float32)]
    converter = tf.lite.TFLiteConverter.from_saved_model(str(tmp))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(converter.convert())
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
