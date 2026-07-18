import json
import time
import numpy as np
import paho.mqtt.client as mqtt

BINS = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)

scores = []

def dist(scores):
    counts, _ = np.histogram(scores, bins=BINS)
    p = counts.astype(np.float64) / max(1, counts.sum())
    return np.maximum(p, 1e-6)

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    scores.append(float(payload["confidence"]))

client = mqtt.Client()
client.on_message = on_message

client.connect("localhost", 1883, 60)
client.subscribe("logibridge/trucks/TRUCK_001/inference")
client.loop_start()

print("Collecting 300 confidence scores...")

while len(scores) < 300:
    print(f"\rCollected {len(scores)}/300", end="", flush=True)
    time.sleep(0.2)

client.loop_stop()

reference = {
    "bins": BINS.tolist(),
    "distribution": dist(scores).tolist(),
    "source": "300 normal inference outputs"
}

with open("reference_dist.json", "w") as f:
    json.dump(reference, f, indent=2)

print("\nReference saved successfully!")
print("Distribution:", reference["distribution"])