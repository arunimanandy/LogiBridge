#!/usr/bin/env python3
"""Preprocessing for LogiEdge: filtering, sliding-window features, normalization."""
import argparse
from collections import deque
import numpy as np
from scipy.stats import kurtosis

FEATURE_NAMES = [
    "temp_mean", "temp_std", "temp_rate_c_per_min",
    "vibration_rms", "vibration_peak", "vibration_kurtosis"
]

class MovingAverage:
    def __init__(self, n=5):
        self.values = deque(maxlen=n)
    def update(self, x):
        self.values.append(float(x))
        return float(np.mean(self.values))

class WindowFeatureExtractor:
    def __init__(self, window_seconds=30, step_seconds=10):
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds
        self.temp = []
        self.vib = []
        self.last_emit = None
    def add_temperature(self, ts, value):
        self.temp.append((float(ts), float(value)))
    def add_vibration(self, ts, value):
        self.vib.append((float(ts), float(value)))
    def _trim(self, now):
        cutoff = now - self.window_seconds
        self.temp = [(t,v) for t,v in self.temp if t >= cutoff]
        self.vib = [(t,v) for t,v in self.vib if t >= cutoff]
    def ready(self, now):
        if self.last_emit is None:
            self.last_emit = now
            return False
        return now - self.last_emit >= self.step_seconds
    def extract_if_ready(self, now):
        self._trim(now)
        if not self.ready(now):
            return None
        if len(self.temp) < 5 or len(self.vib) < 3:
            return None
        self.last_emit = now
        temps = np.array([v for _,v in self.temp], dtype=np.float32)
        vibs = np.array([v for _,v in self.vib], dtype=np.float32)
        rate = (temps[-1] - temps[0]) / max(1e-6, (self.temp[-1][0] - self.temp[0][0])) * 60.0
        vib_kurt = float(kurtosis(vibs, fisher=False, bias=False)) if len(vibs) >= 4 else 3.0
        return np.array([
            temps.mean(), temps.std(ddof=0), rate,
            np.sqrt(np.mean(vibs**2)), vibs.max(), vib_kurt
        ], dtype=np.float32)

def normalize_features(x, stats_path):
    stats = np.load(stats_path, allow_pickle=True).item()
    mean = stats["mean"].astype(np.float32)
    std = stats["std"].astype(np.float32)
    return (x - mean) / np.maximum(std, 1e-6)

def simulate_feature_matrix(minutes=10, anomaly="none", seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for start in range(0, minutes*60 - 30 + 1, 10):
        temp = rng.normal(4.0, 0.3, 30)
        vib = rng.normal(0.45, 0.05, 15)
        if anomaly in ("temp_drift", "combined"):
            temp += 0.08 * np.arange(30)
        if anomaly in ("vibration", "combined"):
            vib = rng.normal(1.2, 0.15, 15)
        rate = (temp[-1] - temp[0]) / 29.0 * 60.0
        rows.append([temp.mean(), temp.std(ddof=0), rate, np.sqrt(np.mean(vib**2)), vib.max(), kurtosis(vib, fisher=False, bias=False)])
    return np.array(rows, dtype=np.float32)

def make_stats(out):
    clean = simulate_feature_matrix(minutes=10, anomaly="none")
    stats = {"feature_names": FEATURE_NAMES, "mean": clean.mean(axis=0), "std": clean.std(axis=0) + 1e-6}
    np.save(out, stats)
    print(f"Saved normalization stats to {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-stats", action="store_true")
    parser.add_argument("--out", default="training_stats.npy")
    args = parser.parse_args()
    if args.make_stats:
        make_stats(args.out)
