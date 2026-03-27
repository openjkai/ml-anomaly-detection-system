"""
Generate synthetic operational metrics with daily/weekly seasonality and injected anomalies.

Output: data/raw/metrics.csv (schema matches project spec).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS, RANDOM_SEED, RAW_METRICS_CSV
from utils import set_random_seed


def _hours_since_start(n: int, step_minutes: int = 5) -> np.ndarray:
    return np.arange(n, dtype=np.float64) * (step_minutes / 60.0)


def build_baseline(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Realistic baseline metrics with daily + weekly cycles and noise."""
    h = _hours_since_start(n)
    day = 2 * np.pi * h / 24.0
    week = 2 * np.pi * h / (24.0 * 7.0)

    cpu = 38.0 + 14.0 * np.sin(day) + 5.0 * np.sin(week + 0.3) + rng.normal(0.0, 2.5, n)
    cpu = np.clip(cpu, 5.0, 85.0)

    mem = 46.0 + 9.0 * np.sin(day + 0.8) + 3.5 * np.sin(week) + rng.normal(0.0, 1.8, n)
    mem = np.clip(mem, 10.0, 92.0)

    latency = (
        95.0
        + 35.0 * np.sin(day + 0.2)
        + 12.0 * np.sin(week + 1.1)
        + np.abs(rng.normal(0.0, 8.0, n))
    )
    latency = np.clip(latency, 45.0, 420.0)

    err = (
        0.012
        + 0.006 * np.sin(day + 1.5)
        + 0.003 * np.sin(week)
        + rng.uniform(-0.004, 0.004, n)
    )
    err = np.clip(err, 0.0, 0.08)

    req = (
        520.0
        + 160.0 * np.sin(day - 0.5)
        + 90.0 * np.sin(week + 0.6)
        + rng.normal(0.0, 35.0, n)
    )
    req = np.maximum(req, 80.0)

    disk = 28.0 + 11.0 * np.sin(day + 2.0) + rng.normal(0.0, 4.0, n)
    disk = np.clip(disk, 5.0, 120.0)

    net = (
        165.0
        + 48.0 * np.sin(day + 0.4)
        + 22.0 * np.sin(week + 2.2)
        + rng.normal(0.0, 18.0, n)
    )
    net = np.maximum(net, 10.0)

    return {
        "cpu_usage": cpu,
        "memory_usage": mem,
        "request_latency_ms": latency,
        "error_rate": err,
        "request_count": req,
        "disk_io": disk,
        "network_in_mb": net,
    }


def _hour_of_day(idx: int) -> float:
    steps_per_day = (24 * 60) // 5
    return (idx % steps_per_day) * (5.0 / 60.0)


def inject_anomalies(
    data: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Inject labeled anomaly segments. Returns is_anomaly boolean array.
    """
    n = len(next(iter(data.values())))
    is_anomaly = np.zeros(n, dtype=np.int8)

    def mark(start: int, end: int) -> None:
        is_anomaly[start:end] = 1

    # Precompute business-hours mask (roughly 09:00–17:59 local series time)
    hod = np.array([_hour_of_day(i) for i in range(n)])
    business = (hod >= 9.0) & (hod < 18.0)

    # Random segment placements (non-overlapping preferred: track used ranges)
    used: list[tuple[int, int]] = []

    def overlaps(a: int, b: int) -> bool:
        for u, v in used:
            if not (b <= u or a >= v):
                return True
        return False

    def reserve(a: int, b: int) -> bool:
        if a < 0 or b > n or a >= b:
            return False
        if overlaps(a, b):
            return False
        used.append((a, b))
        return True

    scenarios = [
        "cpu_spike",
        "memory_leak",
        "latency_spike",
        "error_burst",
        "traffic_crash_business",
        "disk_network_surge",
        "combo_api_incident",
    ]

    for _ in range(rng.integers(18, 28)):
        length = int(rng.integers(2, 8))
        start = int(rng.integers(50, max(51, n - length - 50)))
        end = start + length
        if not reserve(start, end):
            continue
        kind = scenarios[int(rng.integers(0, len(scenarios)))]

        if kind == "cpu_spike":
            data["cpu_usage"][start:end] = rng.uniform(91.0, 97.0, size=end - start)
            data["request_latency_ms"][start:end] += rng.uniform(
                40.0, 120.0, size=end - start
            )
        elif kind == "memory_leak":
            ramp = np.linspace(0.0, 1.0, end - start)
            data["memory_usage"][start:end] = np.clip(
                data["memory_usage"][start:end] + 25.0 + 40.0 * ramp,
                0.0,
                99.0,
            )
        elif kind == "latency_spike":
            data["request_latency_ms"][start:end] = rng.uniform(
                850.0, 1400.0, size=end - start
            )
        elif kind == "error_burst":
            data["error_rate"][start:end] = rng.uniform(0.18, 0.38, size=end - start)
        elif kind == "traffic_crash_business":
            seg_idx = np.arange(start, end)
            bh = business[seg_idx]
            if bh.any():
                data["request_count"][seg_idx[bh]] = rng.uniform(
                    15.0, 55.0, size=int(bh.sum())
                )
        elif kind == "disk_network_surge":
            data["disk_io"][start:end] = rng.uniform(85.0, 115.0, size=end - start)
            data["network_in_mb"][start:end] = rng.uniform(8.0, 35.0, size=end - start)
        else:  # combo_api_incident
            data["cpu_usage"][start:end] = rng.uniform(82.0, 94.0, size=end - start)
            data["memory_usage"][start:end] = rng.uniform(78.0, 92.0, size=end - start)
            data["request_latency_ms"][start:end] = rng.uniform(
                700.0, 1100.0, size=end - start
            )
            data["error_rate"][start:end] = rng.uniform(0.12, 0.30, size=end - start)

        mark(start, end)

    return is_anomaly


def generate_dataframe(
    days: int = 30,
    step_minutes: int = 5,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Create timestamps, baseline metrics, anomalies, and labels."""
    rng = np.random.default_rng(seed)
    set_random_seed(seed)
    n = int((days * 24 * 60) // step_minutes)
    idx = pd.date_range(
        start="2026-01-01 00:00:00",
        periods=n,
        freq=f"{step_minutes}min",
    )
    base = build_baseline(n, rng)
    is_anomaly = inject_anomalies(base, rng)

    df = pd.DataFrame(base)
    df.insert(0, "timestamp", idx)
    df["is_anomaly"] = is_anomaly
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic metrics.csv")
    parser.add_argument(
        "--days", type=int, default=30, help="Number of days of history"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RAW_METRICS_CSV,
        help="Output CSV path",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    df = generate_dataframe(days=args.days, seed=args.seed)
    # Column order per spec
    cols = ["timestamp", *FEATURE_COLUMNS, "is_anomaly"]
    df = df[cols]
    df.to_csv(out, index=False, float_format="%.4f")

    n_pos = int(df["is_anomaly"].sum())
    print(f"Wrote {len(df)} rows to {out}")
    print(f"Labeled anomalies: {n_pos} ({100 * n_pos / len(df):.2f}%)")


if __name__ == "__main__":
    main()
