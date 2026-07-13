# Hardware Selection and Roofline Analysis

## Constraint Triangle

Recommended option: **Raspberry Pi 5 8 GB + AI HAT+**.

| Option | Power | Pilot Cost | Full Fleet Cost | Decision |
|---|---:|---:|---:|---|
| Raspberry Pi 5 + AI HAT+ | 7.5 W | ₹12.75 lakh | ₹39.75 lakh | Recommended |
| Jetson Orin Nano Super | 15 W | ₹38.25 lakh | ₹119.25 lakh | Reject: power and cost |
| STM32H7 custom MCU | 0.4 W | ₹2.98 lakh | ₹9.28 lakh | Reject: too constrained for Docker/MLOps |

The dominant deployment vertex is balanced **latency + power + fleet cost**. The selected hardware is within the 10 W AI budget, supports Linux, Docker, MQTT, local logging, TFLite/Hailo acceleration, and field maintainability.

## Arithmetic Intensity and Roofline

- Model FLOPs per inference: 45 MFLOPs.
- Data accessed per inference: 18 MB.
- Arithmetic Intensity = 45 MFLOPs / 18 MB = **2.5 FLOP/byte**.
- Raspberry Pi 5 CPU ridge point = 16 GFLOP/s / 12 GB/s = **1.33 FLOP/byte**.
- Since 2.5 > 1.33, the model is **compute-bound** under this CPU roofline estimate.

Optimisation should reduce compute operations and improve execution efficiency: INT8 quantisation, pruning, vectorisation, or accelerator offload.
