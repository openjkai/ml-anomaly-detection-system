# ML Anomaly Detection System for Production Monitoring

Machine learning project that flags abnormal behavior in operational time-series metrics (CPU, memory, latency, errors, throughput, disk, network) using **Isolation Forest** and a **TensorFlow autoencoder** (planned).

## Features (in progress)

- Synthetic observability dataset with daily/weekly seasonality and injected anomalies
- Modular `src/` layout for preprocessing, training, and evaluation

## Quick start

```bash
cd ml-anomaly-detection-system
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/generate_data.py
```

This writes `data/raw/metrics.csv` (gitignored by default; regenerate anytime).

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
