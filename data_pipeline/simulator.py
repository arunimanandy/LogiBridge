#!/usr/bin/env python3
"""Cold-chain truck sensor simulator for LogiEdge.

Publishes three streams to local Mosquitto:
- temperature: 1 Hz
- vibration_rms: 0.5 Hz derived RMS stream
- door_event: discrete OPEN/CLOSE events
"""
import argparse, json, random, signal, time
from datetime import datetime, timezone
import numpy as np

try:
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None

RUNNING = True

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def stop_handler(*_):
    global RUNNING
    RUNNING = False

def temperature_value(step, anomaly):
    base = np.random.normal(4.0, 0.3)
    if anomaly in ("temp_drift", "combined"):
        return base + 0.08 * step
    return base

def vibration_value(anomaly):
    if anomaly in ("vibration", "combined"):
        return max(0.0, np.random.normal(1.2, 0.15))
    return max(0.0, np.random.normal(0.45, 0.05))

def publish(client, topic, payload, dry_run=False):
    msg = json.dumps(payload)
    if dry_run or client is None:
        print(topic, msg, flush=True)
    else:
        client.publish(topic, msg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anomaly", choices=["none", "temp_drift", "vibration", "combined"], default="none")
    parser.add_argument("--truck-id", default="TRUCK_001")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--dry-run", action="store_true", help="Print messages instead of publishing to MQTT")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    client = None
    if not args.dry_run:
        if mqtt is None:
            raise RuntimeError("paho-mqtt is not installed. Run pip install -r requirements.txt")
        client = mqtt.Client()
        client.connect(args.host, args.port, 60)
        client.loop_start()

    prefix = f"logibridge/trucks/{args.truck_id}"
    step = 0
    last_vib = 0.0
    last_door = time.time()
    door_state = "CLOSE"

    while RUNNING:
        ts = now_iso()
        temp = float(temperature_value(step, args.anomaly))
        publish(client, f"{prefix}/sensors/temperature", {"ts": ts, "value": temp, "unit": "C"}, args.dry_run)

        if time.time() - last_vib >= 2.0:
            vib = float(vibration_value(args.anomaly))
            publish(client, f"{prefix}/sensors/vibration_rms", {"ts": ts, "value": vib, "unit": "g"}, args.dry_run)
            last_vib = time.time()

        if time.time() - last_door >= random.randint(90, 180):
            door_state = "OPEN" if door_state == "CLOSE" else "CLOSE"
            publish(client, f"{prefix}/sensors/door_event", {"ts": ts, "event": door_state}, args.dry_run)
            last_door = time.time()

        step += 1
        time.sleep(1.0)

    if client:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()
