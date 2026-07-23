import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--stats", required=True)
args = parser.parse_args()

# Load dataset
data = np.load(args.data)
X = data["X"].astype(np.float32)
y = data["y"].astype(np.int64)

_, X_val, _, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load normalization stats
stats = np.load(args.stats, allow_pickle=True).item()
mean = stats["mean"]
std = stats["std"]

# Normalize
X_val = (X_val - mean) / std
print("Using:", args.stats)
print("First normalized sample:", X_val[0])

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=args.model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

scale, zero = input_details[0]["quantization"]

predictions = []

for sample in X_val:
    x = sample.reshape(1, -1)

    if input_details[0]["dtype"] == np.int8:
        x = np.round(x / scale + zero).astype(np.int8)

    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])

    pred = np.argmax(output)
    predictions.append(pred)

acc = accuracy_score(y_val, predictions)

print(f"Validation Accuracy: {acc:.4f}")