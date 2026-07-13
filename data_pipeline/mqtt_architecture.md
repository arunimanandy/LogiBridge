# MQTT Architecture

| Topic | QoS | Purpose |
|---|---:|---|
| `logibridge/trucks/{truck_id}/sensors/temperature` | 0 | High-frequency local temperature stream |
| `logibridge/trucks/{truck_id}/sensors/vibration_rms` | 0 | Derived vibration stream |
| `logibridge/trucks/{truck_id}/sensors/door_event` | 1 | Door open/close event |
| `logibridge/trucks/{truck_id}/features/window_30s` | 0 | Internal feature vector stream |
| `logibridge/trucks/{truck_id}/inference` | 1 | Class, label, confidence, timestamp |
| `logibridge/trucks/{truck_id}/alerts` | 2 | Warning/Critical alert escalation |
| `logibridge/trucks/{truck_id}/mlops/psi` | 1 | Drift monitoring value |
| `logibridge/fleet/{truck_id}/sync/alerts` | 1 | Store-and-forward sync to backend |

QoS 0 is suitable for high-frequency streams where stale samples can be dropped. QoS 1 is used for inference and MLOps messages. QoS 2 is reserved for safety-critical alert messages.
