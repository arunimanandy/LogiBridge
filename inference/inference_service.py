#!/usr/bin/env python3
"""MQTT inference service: preprocessing -> TFLite inference -> MQTT publish."""
import argparse, json, os, time
from collections import deque
import numpy as np
import paho.mqtt.client as mqtt

try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except Exception:
    from tflite_runtime.interpreter import Interpreter

LABELS = {0: "Normal", 1: "Warning", 2: "Critical"}

def load_stats(path):
    return np.load(path, allow_pickle=True).item()

def run_tflite(interpreter, x):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    qx = x.astype(np.float32)
    if input_details["dtype"] == np.int8:
        scale, zero = input_details["quantization"]
        qx = np.clip(qx / scale + zero, -128, 127).astype(np.int8)
    interpreter.set_tensor(input_details["index"], qx.reshape(1, -1))
    interpreter.invoke()
    out = interpreter.get_tensor(output_details["index"])[0]
    if output_details["dtype"] == np.int8:
        scale, zero = output_details["quantization"]
        out = scale * (out.astype(np.float32) - zero)
    exp = np.exp(out - np.max(out))
    return exp / np.sum(exp)

class Service:
    def __init__(self, args):
        self.args = args
        self.stats = load_stats(args.stats_path)
        self.temp = deque(maxlen=30)
        self.vib = deque(maxlen=15)
        self.interpreter = Interpreter(model_path=args.model_path)
        self.interpreter.allocate_tensors()
        self.client = mqtt.Client()
        self.prefix = f"logibridge/trucks/{args.truck_id}"
    def on_message(self, client, userdata, msg):
        payload = json.loads(msg.payload.decode())
        now = time.time()
        if msg.topic.endswith("temperature"):
            self.temp.append(float(payload["value"]))
        elif msg.topic.endswith("vibration_rms"):
            self.vib.append(float(payload["value"]))
        if len(self.temp) >= 30 and len(self.vib) >= 15:
            self.infer(now)
    def infer(self, now):
        temp = np.array(self.temp, dtype=np.float32)
        vib = np.array(self.vib, dtype=np.float32)
        rate = (temp[-1] - temp[0]) / 29.0 * 60.0
        feats = np.array([temp.mean(), temp.std(), rate, np.sqrt(np.mean(vib**2)), vib.max(), 3.0], dtype=np.float32)
        x = (feats - self.stats["mean"]) / np.maximum(self.stats["std"], 1e-6)
        probs = run_tflite(self.interpreter, x)
        cls = int(np.argmax(probs)); conf = float(np.max(probs))
        result = {"truck_id": self.args.truck_id, "ts": now, "class": cls, "label": LABELS[cls], "confidence": conf}
        self.client.publish(f"{self.prefix}/inference", json.dumps(result), qos=1)
        if cls in (1, 2):
            self.client.publish(f"{self.prefix}/alerts", json.dumps(result), qos=2)
        print(result, flush=True)
    def run(self):
        self.client.on_message = self.on_message
        self.client.connect(self.args.host, self.args.port, 60)
        self.client.subscribe(f"{self.prefix}/sensors/temperature")
        self.client.subscribe(f"{self.prefix}/sensors/vibration_rms")
        self.client.loop_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--truck-id", default=os.getenv("TRUCK_ID", "TRUCK_001"))
    parser.add_argument("--model-path", default=os.getenv("MODEL_PATH", "model.tflite"))
    parser.add_argument("--stats-path", default=os.getenv("STATS_PATH", "training_stats.npy"))
    parser.add_argument("--host", default=os.getenv("MQTT_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    Service(parser.parse_args()).run()
