# ML Anomaly Detection System for Production Monitoring

Machine learning project that flags abnormal behavior in operational time-series metrics (CPU, memory, latency, errors, throughput, disk, network) using **Isolation Forest** and a **TensorFlow autoencoder** (planned).

## Features (in progress)

- Synthetic observability dataset with daily/weekly seasonality and injected anomalies
- Preprocessing: sort by time, drop duplicate timestamps, fill missing values, chronological train/test split, `StandardScaler` fit on train only (saved as `models/scaler.pkl`)
- Modular `src/` layout for training and evaluation (next)

## Quick start

```bash
cd ml-anomaly-detection-system
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/generate_data.py
python src/preprocess.py
```

This writes `data/raw/metrics.csv`, then `data/processed/train.csv`, `data/processed/test.csv`, and `models/scaler.pkl` (artifacts are gitignored by default; regenerate anytime).

**Preprocessing notes:** CSVs keep **unscaled** feature columns; training scripts should `joblib.load` the scaler and transform `FEATURE_COLUMNS` before fitting models.

## Project layout

```text
Metrics → Preprocessing → Model → Anomaly Score → Alert
```

See the repo tree: `data/`, `notebooks/`, `src/`, `models/`, `outputs/`, `tests/`.

## Dataset columns

| Column | Description |
|--------|-------------|
| `timestamp` | Observation time |
| `cpu_usage` | CPU utilization % |
| `memory_usage` | Memory utilization % |
| `request_latency_ms` | Average request latency |
| `error_rate` | Proportion of failed requests |
| `request_count` | Requests in the interval |
| `disk_io` | Disk activity |
| `network_in_mb` | Incoming traffic |
| `is_anomaly` | Injected label for evaluation |

## License

MIT (add a `LICENSE` file when you publish).
