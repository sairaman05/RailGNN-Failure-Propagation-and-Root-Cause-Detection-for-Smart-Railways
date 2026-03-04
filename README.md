# 🚆 Graph-Based Failure Propagation and Root Cause Detection for Smart Railway Infrastructure

> **Using Kafka Streaming and Airflow-Orchestrated Temporal GNN**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Kafka](https://img.shields.io/badge/Apache_Kafka-7.5-black?logo=apachekafka)](https://kafka.apache.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-orange?logo=pytorch)](https://pytorch-geometric.readthedocs.io)
[![Airflow](https://img.shields.io/badge/Apache_Airflow-2.x-017CEE?logo=apacheairflow)](https://airflow.apache.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Overview

Traditional railway maintenance relies on fixed inspection schedules that miss developing faults between cycles. This project builds a **predictive maintenance system** that:

- Models the railway network as a **graph** where nodes are components (tracks, switches, signals, bridges) and edges are physical/electrical/load dependencies
- Streams live sensor telemetry through **Apache Kafka**
- Applies a **Temporal Graph Neural Network (TGNN)** to predict how failures propagate across interconnected nodes
- Identifies **root cause components** that initiate failure cascades
- Automates retraining and drift detection via **Apache Airflow DAGs**
- Visualises everything on a live **Streamlit dashboard**

---

## 🏗️ Architecture

```
[Phase 1]  sensor_simulator.py
     │
     ▼  railway_sensor_data.csv (100K rows)
[Phase 2]  topology.py + railway_graph.py
     │
     ▼  PyG graph object (20 nodes, 70 edges, 16-dim features)
[Phase 3]  producer.py ──► [Kafka: railway.sensors.raw]
                                  │
                           consumer.py + feature_engine.py
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                        ▼
 [railway.sensors.processed]               [railway.alerts]
              │
[Phase 4]  tgnn.py (GCN + GRU)
     │
     ▼  risk scores + root cause attribution
[Phase 5]  Airflow DAGs (retrain · evaluate · drift)
     │
     ▼  champion model checkpoint
[Phase 6]  Streamlit dashboard (live graph + alerts)
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data Generation | Python, NumPy, Pandas | 100K sensor records with degradation patterns |
| Streaming | Apache Kafka 7.5 | Real-time ingestion with ordering guarantees |
| Stream Processing | Kafka Consumer + Python | Rolling statistics and feature engineering |
| Graph Modelling | NetworkX, PyTorch Geometric | Railway topology and dynamic node features |
| Deep Learning | PyTorch, PyG Temporal | Temporal GNN for failure prediction |
| Orchestration | Apache Airflow | DAGs for retraining, evaluation, deployment |
| Dashboard | Streamlit, Plotly | Live visualisation with graph rendering |
| Storage | CSV / Parquet, SQLite | Historical data and model metadata |
| Containerisation | Docker Compose | Reproducible Kafka + Zookeeper setup |

---

## 📁 Project Structure

```
railway-failure-detection/
├── data/
│   ├── raw/
│   │   ├── railway_sensor_data.csv        # 100K sensor records (Phase 1 output)
│   │   └── dataset_summary.json
│   └── processed/
│       ├── processed_features_sample.json  # rolling features (Phase 3 output)
│       ├── alerts.json
│       └── phase3_summary.json
├── src/
│   ├── data_generation/                   # Phase 1
│   │   ├── config.py
│   │   ├── degradation_patterns.py
│   │   └── sensor_simulator.py
│   ├── graph/                             # Phase 2
│   │   ├── topology.py
│   │   ├── node_features.py
│   │   ├── railway_graph.py
│   │   └── visualize.py
│   ├── kafka/                             # Phase 3
│   │   ├── config.py
│   │   ├── producer.py
│   │   ├── consumer.py
│   │   ├── feature_engine.py
│   │   ├── alert_manager.py
│   │   └── simulate.py
│   ├── model/                             # Phase 4 (upcoming)
│   ├── dashboard/                         # Phase 6 (upcoming)
│   └── utils/
├── airflow/
│   └── dags/                              # Phase 5 (upcoming)
├── test_graph.py                          # Phase 2 tests
├── test_phase3.py                         # Phase 3 tests
├── docker-compose.yaml
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Docker + Docker Compose v2 ([upgrade guide](#-docker-setup))
- 4 GB RAM minimum (8 GB recommended for model training)

### 1. Clone and set up environment

```bash
git clone https://github.com/your-username/railway-failure-detection.git
cd railway-failure-detection

python3 -m venv venv
source venv/bin/activate

pip install numpy pandas networkx torch torch-geometric kafka-python streamlit plotly
```

### 2. Phase 1 — Generate sensor data

```bash
python -m src.data_generation.sensor_simulator
```

Generates `data/raw/railway_sensor_data.csv` — 100,000 sensor records across 20 components (~42 hours simulated time) with built-in degradation events.

### 3. Phase 2 — Build the graph

```bash
python test_graph.py
xdg-open data/processed/railway_graph.html   # view topology
```

### 4. Phase 3 — Stream processing (offline, no Kafka needed)

```bash
python -m src.kafka.simulate --limit 10000
python test_phase3.py
```

### 4b. Phase 3 — Live Kafka mode

```bash
# Terminal 0 — start Kafka
docker-compose up -d

# Terminal A — stream CSV to Kafka
python -m src.kafka.producer --speed 10

# Terminal B — consume and process
python -m src.kafka.consumer --every 500

# Kafka UI — http://localhost:8080
```

---

## 📊 Dataset Details

| Metric | Value |
|---|---|
| Total records | 100,000 |
| Components | 20 (10 tracks, 4 switches, 3 signals, 3 bridges) |
| Sensor channels | vibration, temperature, load, current |
| Simulated time span | ~42 hours |
| Normal risk | 63.4% |
| Low risk | 29.1% |
| Medium risk | 7.5% |
| High risk | 0.04% |

**Built-in degradation events:**

| Dataset Region | Component(s) | Failure Mode |
|---|---|---|
| Rows 30k–50k | Track T05 | Mechanical wear — exponential vibration curve |
| Rows 50k–60k | SW3, SG2, T04, T06 | Propagation from T05 overheating |
| Rows 80k–100k | BR2, SW4, T07 | Bridge overheating → electrical fault → track propagation |

---

## 🔬 Phase Details

### ✅ Phase 1 — Synthetic Data Generation
Simulates 20 railway components with realistic sensor noise, seasonal baselines, and phased degradation/propagation events.

**Key files:** `src/data_generation/sensor_simulator.py`, `degradation_patterns.py`, `config.py`

---

### ✅ Phase 2 — Graph Modelling
Defines the railway network as a PyTorch Geometric compatible graph with 20 nodes, 70 directed edges, and 16-dimensional node feature vectors.

**Edge types:** physical · electrical · load-sharing · signal-control

**Node features:** raw sensors (4) + rolling stats (4) + trend signals (2) + component type one-hot (4) + extras (2)

**Key files:** `src/graph/topology.py`, `node_features.py`, `railway_graph.py`, `visualize.py`

---

### ✅ Phase 3 — Kafka Streaming & Feature Engineering
Streams sensor records through Kafka, enriches each record with 30+ rolling features per component, and fires structured alerts with cooldown deduplication.

**Kafka topics:**
- `railway.sensors.raw` — raw producer output
- `railway.sensors.processed` — enriched feature records
- `railway.alerts` — structured alert events

**Alert levels:** CRITICAL / HIGH / MEDIUM / LOW (with configurable thresholds and cooldown windows)

**Key files:** `src/kafka/producer.py`, `consumer.py`, `feature_engine.py`, `alert_manager.py`, `simulate.py`

---

### 🔜 Phase 4 — Temporal GNN
GCN + GRU architecture operating on temporal node feature tensors of shape `(T=12, N=20, F=16)` for per-node risk score regression and root cause attribution via attention weights.

---

### 🔜 Phase 5 — Airflow Orchestration
DAGs for daily retraining, weekly evaluation, hourly drift detection, and on-trigger model promotion.

---

### 🔜 Phase 6 — Live Dashboard
Streamlit app with interactive network graph, live alert feed, component drill-down, propagation path visualisation, and model metric tracking.

---

## 🐳 Docker Setup

If `docker-compose --version` shows v1.x.x, upgrade to v2:

```bash
sudo curl -SL https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-linux-x86_64 \
  -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo apt remove docker-compose   # remove the old Python version
sudo systemctl start docker
```

---


## TO run uptil NOW

# Terminal 0 — start Kafka
docker-compose up -d

# Terminal A — stream CSV to Kafka
python -m src.kafka.producer --speed 10

# Terminal B — consume and process
python -m src.kafka.consumer --every 500

# Browser — Kafka UI
http://localhost:8080

do these commands

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

