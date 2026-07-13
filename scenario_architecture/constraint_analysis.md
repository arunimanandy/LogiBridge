# Constraint Analysis — LogiEdge

## Latency
A refrigeration failure can raise cargo temperature by 1°C/minute, so LogiEdge must detect and alert within 90 seconds of a fault signature. A cloud-only design is not acceptable because rural cellular latency and outages can exceed the allowed detection window. LogiEdge performs inference on the truck, publishes the result locally, writes alerts to a local log, and syncs when connectivity returns.

## Bandwidth
Assuming numeric payloads with an 8-byte timestamp and float32 sensor values:

- Temperature: 1 Hz × 86,400 seconds × 12 bytes ≈ 1.04 MB/day.
- Vibration raw 3-axis: 500 Hz × 86,400 seconds × 20 bytes ≈ 864 MB/day.
- Total raw stream ≈ 865 MB/truck/day, excluding JSON overhead.
- Transmission cost at ₹0.10/MB ≈ ₹86.50/truck/day or about ₹7,352/day for 85 trucks.

Edge processing transmits alerts, confidence, summaries, and synced logs rather than raw vibration streams.

## Connectivity
The route has cellular gaps of 35–90 minutes. During those gaps, a cloud-only system cannot ingest sensor data or return inference decisions. LogiEdge continues to run MQTT, preprocessing, inference, local alerting, local logging, and PSI monitoring on the truck.

## Privacy
On-device inference reduces third-party exposure of pharmaceutical cargo condition data. The operations centre receives only necessary alerts and synchronized logs, while raw high-frequency sensor streams remain local unless explicitly extracted for audit.
