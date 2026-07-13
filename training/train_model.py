#!/usr/bin/env python3
"""Train 2-hidden-layer MLP and save validation metrics."""
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score
import tensorflow as tf

def build_model(input_dim=6):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax")
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training/logiedge_dataset.npz")
    parser.add_argument("--out", default="training/models")
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()

    data = np.load(args.data)
    X, y = data["X"].astype(np.float32), data["y"].astype(np.int64)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    model = build_model(X.shape[1])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs, callbacks=[cb], verbose=2)

    pred = np.argmax(model.predict(X_val), axis=1)
    acc = float(accuracy_score(y_val, pred))
    recalls = recall_score(y_val, pred, average=None, labels=[0,1,2]).tolist()
    metrics = {
        "validation_accuracy": acc,
        "class_recalls": {"0_Normal": recalls[0], "1_Warning": recalls[1], "2_Critical": recalls[2]},
        "confusion_matrix": confusion_matrix(y_val, pred).tolist(),
        "classification_report": classification_report(y_val, pred, output_dict=True)
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.export(out / "fp32_saved_model")
    model.save(out / "model_fp32.keras")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    if acc < 0.88:
        raise SystemExit("Validation accuracy below 88%; revisit features/model before proceeding.")

if __name__ == "__main__":
    main()
