#!/usr/bin/env python3
"""PSI drift monitor for output confidence distribution."""
import argparse, json, time
from collections import deque
import numpy as np
import paho.mqtt.client as mqtt

BINS = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)

def dist(scores):
    counts, _ = np.histogram(scores, bins=BINS)
    p = counts.astype(np.float64) / max(1, counts.sum())
    return np.maximum(p, 1e-6)

def psi(expected, actual):
    expected = np.maximum(np.array(expected, dtype=np.float64), 1e-6)
    actual = np.maximum(np.array(actual, dtype=np.float64), 1e-6)
    return float(np.sum((actual - expected) * np.log(actual / expected)))

class Monitor:
    def __init__(self, args):
        self.args = args
        self.scores = deque(maxlen=100)
        self.ref = json.loads(open(args.reference).read())["distribution"]
        self.client = mqtt.Client()
    def on_message(self, client, userdata, msg):
        payload = json.loads(msg.payload.decode())
        self.scores.append(float(payload["confidence"]))
    def run(self):
        prefix = f"logibridge/trucks/{self.args.truck_id}"
        self.client.on_message = self.on_message
        self.client.connect(self.args.host, self.args.port, 60)
        self.client.subscribe(f"{prefix}/inference", qos=1)
        self.client.loop_start()
        while True:
            time.sleep(60)
            if len(self.scores) < 20:
                print(f"PSI waiting for samples: {len(self.scores)}/100", flush=True)
                continue
            cur = dist(list(self.scores))
            value = psi(self.ref, cur)
            print(f"Current PSI={value:.3f}", flush=True)
            self.client.publish(f"{prefix}/mlops/psi", json.dumps({"psi": value, "n": len(self.scores)}), qos=1)
            if value > 0.25:
                print(f"[LOGIBRIDGE DRIFT ALERT] PSI={value:.3f}", flush=True)

def make_reference(out):
    # Initial scaffold reference: high-confidence normal outputs.
    scores = np.clip(np.random.default_rng(42).normal(0.92, 0.04, 300), 0, 1)
    obj = {"bins": BINS.tolist(), "distribution": dist(scores).tolist(), "source": "replace with 300 clean Normal-class inference outputs"}
    open(out, "w").write(json.dumps(obj, indent=2))
    print(f"Wrote {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--truck-id", default="TRUCK_001")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--reference", default="monitoring/reference_dist.json")
    parser.add_argument("--make-reference", action="store_true")
    args = parser.parse_args()
    if args.make_reference:
        make_reference(args.reference)
    else:
        Monitor(args).run()
