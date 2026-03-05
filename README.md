# 🚆 Graph-Based Failure Propagation and Root Cause Detection for Smart Railway Infrastructure

> **Real-time predictive maintenance using Kafka Streaming, Temporal Graph Neural Networks, and Airflow Orchestration**

## 📌 Overview

Traditional railway maintenance is scheduled on fixed intervals — a component is inspected every N days regardless of its actual condition. This approach misses faults that develop between inspections and cannot predict cascading failures where one degrading component places extra stress on its neighbours.

This project builds a **real-time predictive maintenance system** that fundamentally changes that:

- Models the entire railway network as a **graph** — 20 nodes (tracks, switches, signals, bridges) connected by 70 edges representing physical, electrical, load-sharing, and signal-control dependencies
- Streams **live sensor telemetry** through Apache Kafka as individual JSON messages — one per component, every 2 seconds, no batch files
- Applies a custom **Temporal Graph Neural Network (TGNN)** that combines spatial Graph Convolution with temporal GRU layers to understand how sensor patterns evolve across the connected network over time
- Issues **early warnings 6 time steps before** a component reaches high risk — giving maintenance teams actionable lead time before failure
- Identifies the **root cause component** that initiated a failure cascade, using learned edge attention weights
- Provides a live **Streamlit dashboard** with an interactive railway schematic, risk heatmaps, propagation timeline, and component drill-down
- Automates **model retraining and accuracy monitoring** via Apache Airflow DAGs running on weekly schedule

---

## 🏗️ System Architecture

```
╔══════════════════════════════════════════════════════════════════════════╗
║  PHASE 1 — DATA GENERATION                                              ║
║                                                                          ║
║  src/data_generation/sensor_simulator.py                                ║
║    ├── 20 components × 20,000 time steps = 400,000 records              ║
║    ├── 17 overlapping failure cascades across 6 time clusters           ║
║    ├── risk_score derived from degradation STATE (not sensor formula)   ║
║    └── Output: data/raw/railway_sensor_data.csv                         ║
╚══════════════════════════════════════════════════════════════════════════╝
                              ↓
╔══════════════════════════════════════════════════════════════════════════╗
║  PHASE 2 — GRAPH MODELLING                                              ║
║                                                                          ║
║  src/graph/topology.py        → 20 nodes, 70 edges, 4 edge types        ║
║  src/graph/node_features.py   → 16-dim feature vectors per node         ║
║  src/graph/railway_graph.py   → PyTorch-compatible adjacency matrix     ║
╚══════════════════════════════════════════════════════════════════════════╝
                              ↓
╔══════════════════════════════════════════════════════════════════════════╗
║  PHASE 3 — KAFKA STREAMING PIPELINE                                     ║
║                                                                          ║
║  live_sensor_simulator.py                                               ║
║    ├── 1 JSON message per component every 2 seconds                     ║
║    └── auto-injects degradation at t=30s, 60s, 90s, 150s              ║
║                   ↓  Topic: railway.sensors.raw                          ║
║  consumer.py                                                             ║
║    ├── feature_engine  → rolling mean/std/trend/z-score                 ║
║    ├── TGNN inference  → risk score + class + root cause attention      ║
║    ├── alert_manager   → cooldown-deduplicated alerts                   ║
║    └── writes: data/predictions/live_predictions.json                   ║
╚══════════════════════════════════════════════════════════════════════════╝
                              ↓
╔══════════════════════════════════════════════════════════════════════════╗
║  PHASE 4 — TEMPORAL GRAPH NEURAL NETWORK                               ║
║                                                                          ║
║  Input:    (B, T=12, N=20, F=16)  — 12 past steps, all 20 nodes       ║
║  GCN × 2: spatial aggregation — each node sees its graph neighbours    ║
║  GRU × 2: temporal encoding  — 12-step sequence compressed per node   ║
║  Output:   risk_cls (4 classes) + risk_reg (0–1) + root_cause attn    ║
║                                                                          ║
║  KEY DESIGN: Forecasting formulation                                    ║
║    → predicts risk 6 steps into the FUTURE, not current state          ║
║    → model must learn patterns that PRECEDE failures                   ║
║    → prevents data leakage from deterministic label formula            ║
╚══════════════════════════════════════════════════════════════════════════╝
                              ↓
╔══════════════════════════════════════════════════════════════════════════╗
║  PHASE 5 — AIRFLOW ORCHESTRATION (optional)                            ║
║                                                                          ║
║  file_watcher_dag  → every 1 minute  → stream new CSVs from incoming/  ║
║  retrain_dag       → every Sunday    → retrain if accuracy < 0.75      ║
╚══════════════════════════════════════════════════════════════════════════╝
                              ↓
╔══════════════════════════════════════════════════════════════════════════╗
║  PHASE 6 — LIVE DASHBOARD  (http://localhost:8501)                     ║
║                                                                          ║
║  → Railway schematic network graph (risk-coloured nodes/edges)         ║
║  → Root cause attention ranking bar chart                              ║
║  → Risk heatmap over time (all 20 components × last 30 predictions)   ║
║  → Failure propagation grid (component × time step)                   ║
║  → Component drill-down with 50-step trend chart + threshold bands     ║
║  → Per-class model performance metrics                                 ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 📁 Project Structure

```
railway-failure-detection/
│
├── src/
│   │
│   ├── data_generation/
│   │   ├── __init__.py
│   │   ├── config.py                  Component configs, sensor ranges
│   │   ├── degradation_patterns.py    Failure mode definitions
│   │   └── sensor_simulator.py        ★ Main data generator
│   │                                    - 17 failure events, 6 clusters
│   │                                    - risk_score from degradation STATE
│   │                                    - Lower thresholds for balance
│   │                                    Usage: --steps 20000
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── topology.py                Node/edge definitions, 4 edge types
│   │   ├── node_features.py           16-dim feature vector construction
│   │   ├── railway_graph.py           PyTorch-compatible graph object
│   │   └── visualize.py               HTML interactive topology viewer
│   │
│   ├── kafka/
│   │   ├── __init__.py
│   │   ├── config.py                  Topics, thresholds, stream config
│   │   ├── live_sensor_simulator.py   ★ JSON → Kafka every N seconds
│   │                                    - --interval, --degrade, --severity
│   │                                    - Auto fault injection schedule
│   │   ├── consumer.py                ★ Kafka → features → TGNN → alerts
│   │                                    - Warms up over first 24s
│   │                                    - Saves live_predictions.json
│   │   ├── feature_engine.py          Rolling stats per component buffer
│   │   │                                mean, std, trend, z-score, min, max
│   │   ├── alert_manager.py           Level classification + cooldown
│   │   ├── simulate.py                Offline pipeline (no Kafka needed)
│   │   └── producer.py                CSV batch producer (fallback mode)
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── data_loader.py             ★ CSV/JSON → (T,N,F) tensors
│   │                                    - Forecasting formulation (horizon=6)
│   │                                    - Rolling features computed on-the-fly
│   │                                    - risk_score/risk_level NOT in inputs
│   │   ├── tgnn.py                    ★ GCN spatial + GRU temporal
│   │                                    - 196,295 params
│   │                                    - Pure PyTorch, no torch_geometric
│   │   ├── trainer.py                 Training loop + early stopping
│   │                                    - Dynamic class weights from data
│   │                                    - AdamW + ReduceLROnPlateau
│   │   ├── evaluator.py               Accuracy, AUC-ROC, F1, confusion matrix
│   │                                    Usage: --source csv
│   │   └── root_cause.py              Live inference helper
│   │
│   └── dashboard/
│       ├── __init__.py
│       └── app.py                     ★ Streamlit — 7 panels, auto-refresh
│                                        - Network graph (railway schematic)
│                                        - Root cause analysis panel
│                                        - Risk heatmap + propagation grid
│                                        - Component drill-down
│
├── airflow/
│   └── dags/
│       ├── file_watcher_dag.py        Watches data/incoming/ every 1 min
│       └── retrain_dag.py             Weekly retrain if accuracy < 0.75
│
├── data/
│   ├── raw/                           railway_sensor_data.csv  (generated)
│   ├── processed/                     evaluation_results.json
│   │                                  training_log.json
│   │                                  processed_features_sample.json
│   ├── incoming/                      Drop CSVs here for auto-batch mode
│   └── predictions/                   live_predictions.json (dashboard reads)
│
├── checkpoints/                       best_model.pt  (saved after training)
├── test_phase3.py                     Unit tests: feature engine + alerts
├── test_phase4.py                     Unit tests: TGNN + data loader
├── docker-compose.yaml                Kafka + Zookeeper + Kafka UI
├── requirements.txt
└── README.md
```

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data Generation | Python, NumPy, math | Synthetic sensor data with embedded cascade failures |
| Graph Modelling | NetworkX, PyTorch | Railway topology as adjacency matrix |
| Message Streaming | Apache Kafka 7.5 (Confluent) | Real-time high-throughput telemetry ingestion |
| Live Simulation | Python (custom) | JSON messages every 2s with scheduled fault injection |
| Feature Engineering | Python rolling buffers | mean, std, trend, z-score per sensor per component |
| Deep Learning | PyTorch 2.0+ (custom GCN+GRU) | Spatial-temporal graph learning, no torch_geometric |
| Orchestration | Apache Airflow 2.x | DAG scheduling for file watching and retraining |
| Dashboard | Streamlit 1.30+ / Plotly 5.18+ | Interactive network graph, heatmaps, drill-down |
| Infrastructure | Docker Compose v2 | Kafka + Zookeeper + Kafka UI in one command |

---

## 🔬 Technical Deep Dive

### Railway Network Graph

```
20 Nodes — 4 types:
  Tracks   T01–T10   Horizontal spine — main rail segments
  Switches SW1–SW4   Junctions below tracks — route control
  Signals  SG1–SG3   Supervisory nodes at bottom of hierarchy
  Bridges  BR1–BR3   Structural nodes above tracks

70 Edges (bidirectional) — 4 types:
  Physical     Track-to-track adjacency along the rail line
  Structural   Bridge-to-track load-sharing dependency
  Control      Switch-to-track routing dependency
  Supervisory  Signal-to-switch monitoring dependency

Adjacency: SW3 neighbours = {T05, T06, T07, SG2, SG3}
  → When T05 degrades, SW3 sees it in its neighbourhood aggregation
  → This is how the TGNN learns spatial failure propagation
```

### Node Feature Vector — 16 Dimensions

Each of the 20 nodes at each time step has a 16-dimensional feature vector computed **entirely from raw sensor values**. `risk_score` and `risk_level` are deliberately excluded as inputs — they only appear as training labels.

```
Dims  0– 3   Raw sensors          vibration, temperature, load, current
Dims  4– 7   Rolling mean (w=20)  per sensor over last 20 readings
Dims  8–11   Rolling std  (w=20)  variability measure per sensor
Dims 12–15   Trend slope          linear regression slope per sensor

Computed on-the-fly during data loading from raw CSV values.
The rolling window maintains a deque of the last 20 readings per
component per sensor and recomputes stats incrementally.
```

### TGNN Architecture (196,295 parameters)

```
Input tensor:  (B, T=12, N=20, F=16)
                │
                │  Applied at each of the 12 time steps:
                ▼
  ┌─────────────────────────────────────────┐
  │  GraphConvLayer 1   (16 → 64)           │
  │    h_v = σ(W · mean({h_u : u ∈ N(v)})) │
  │    LayerNorm(64) + Dropout(0.3)         │
  ├─────────────────────────────────────────┤
  │  GraphConvLayer 2   (64 → 64)           │
  │    LayerNorm(64) + Dropout(0.3)         │
  └─────────────────────────────────────────┘
                │
                │  Now shape: (B, T=12, N=20, hidden=64)
                │  Each node has seen its 1–2 hop neighbourhood
                ▼
  ┌─────────────────────────────────────────┐
  │  GRU (2 layers, hidden=128, dropout=0.3)│
  │    Input:  sequence of 12 node states   │
  │    Output: final hidden state per node  │
  └─────────────────────────────────────────┘
                │
                │  Now shape: (B, N=20, hidden=128)
                │  Each node summarises its 12-step history
                │  in the context of its graph neighbours
                ▼
      ┌─────────┬──────────┬──────────────┐
      │cls_head │ reg_head │ root_cause   │
      │         │          │ attn         │
      ▼         ▼          ▼
  (B,N,4)    (B,N,1)    (B,N,1)
  risk class  risk score  source prob
  (softmax)   (sigmoid)   (softmax)
```

### Why the Forecasting Formulation

A naive approach predicts **current** risk from **current** sensors. This always produces 100% accuracy because the training data's `risk_level` column was computed from sensor values using a deterministic threshold formula. The model just memorises the formula in epoch 1 — it learns nothing.

```
NAIVE (broken — data leakage):
  Input:  sensor readings at steps [i .. i+11]
  Label:  risk_level AT step i+11        ← same time as last input step
  Result: model memorises f(sensors) = label. acc=1.000 at epoch 1.

FORECASTING (correct):
  Input:  sensor readings at steps [i .. i+11]
  Label:  risk_level AT step i+11+6      ← 6 steps in the FUTURE
  Result: model must learn PRECURSOR PATTERNS to future failure.
          Slowly rising vibration over 12 steps → predicts T05 high risk
          6 steps from now. Genuine learning, not formula memorisation.
```

### Training Loss Function

```python
total_loss = 0.7 × CrossEntropy(predicted_class, future_risk_level)
           + 0.3 × MSE(predicted_score, future_risk_score)
```

Class weights are computed **dynamically** from the actual label distribution of the training dataset:

```python
w_c = total_samples / (n_classes × count_c)
```

With a typical distribution (81% normal, 8% low, 6% medium, 5% high), this produces approximately:

```
normal: 0.31   low: 1.55   medium: 2.16   high: 2.72
```

Each high-risk prediction contributes ~9× more to the loss than a normal one. Without this, the model learns that predicting everything as "normal" achieves 81% accuracy with zero effort — the class weights prevent this collapse.

---

## 🚀 Setup and Installation

### Prerequisites

| Requirement | Version | Check command |
|---|---|---|
| Python | 3.10+ | `python3 --version` |
| Docker + Compose v2 | latest | `docker compose version` |
| CUDA GPU | optional, any | `nvidia-smi` |
| RAM | 8 GB recommended | — |
| Disk space | ~3 GB | — |

### 1 — Clone and install dependencies

```bash
git clone https://github.com/your-username/railway-failure-detection.git
cd railway-failure-detection

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 2 — Generate training data

```bash
python -m src.data_generation.sensor_simulator --steps 20000
```

This creates `data/raw/railway_sensor_data.csv` with:
- **400,000 records** (20,000 time steps × 20 components)
- **17 overlapping failure events** spread across 6 time clusters
- Label distribution: ~81% normal, ~8% low, ~6% medium, ~5% high

Verify the distribution before training:

```bash
python3 -c "
import csv
from collections import Counter
counts = Counter()
with open('data/raw/railway_sensor_data.csv') as f:
    for row in csv.DictReader(f): counts[row['risk_level']] += 1
total = sum(counts.values())
print(f'Total records: {total:,}')
for k, v in counts.items():
    print(f'  {k:8s}: {v:7,}  ({v/total*100:.1f}%)')
"
```

> ⚠️ If this shows `normal: 100%` — an old CSV is present. Delete it and regenerate.

### 3 — Train the TGNN model

```bash
python -m src.model.trainer --source csv --epochs 50 --batch 128 --stride 6
```

**What to expect:**
```
[Trainer] Class weights: normal=0.31  low=1.55  medium=2.16  high=2.72
[Trainer] Device: cuda
Epoch 001/50  train_loss=0.65  acc=0.80  | val_loss=0.68  acc=0.79
Epoch 005/50  train_loss=0.52  acc=0.83  | val_loss=0.54  acc=0.82
...
Epoch 030/50  train_loss=0.38  acc=0.87  | val_loss=0.42  acc=0.85 ✓ best
[Trainer] Checkpoint saved → checkpoints/best_model.pt
```

> ⚠️ If epoch 1 shows `acc=1.000` — this is data leakage. Make sure you are using the latest `sensor_simulator.py` (risk derived from degradation state, not sensor thresholds).

**CLI arguments:**

| Argument | Default | Description |
|---|---|---|
| `--source` | `json` | Use `csv` for training on raw CSV |
| `--epochs` | `50` | Number of training epochs |
| `--batch` | `32` | Batch size — use `128` for faster runs |
| `--stride` | `1` | Sliding window stride — use `6` to reduce sequence count |
| `--lr` | `1e-3` | Adam learning rate |
| `--patience` | `8` | Early stopping patience epochs |
| `--seq-len` | `12` | Input sequence length (time steps) |

### 4 — Evaluate the model

```bash
python -m src.model.evaluator --source csv
```

**Healthy model output:**
```
Overall accuracy : 0.87xx
AUC-ROC (binary) : 0.8x
Per-class metrics:
  Class       Precision    Recall    F1
  normal        0.9xxx     0.9xxx  0.9xxx
  low           0.6xxx     0.5xxx  0.5xxx
  medium        0.5xxx     0.4xxx  0.4xxx
  high          0.7xxx     0.6xxx  0.6xxx

Confusion matrix (rows=true, cols=pred):
        NRM   LOW   MED   HGH
 NRM   XXXX    XX    XX     X   ← mostly correct
 LOW     XX   XXX    XX     X   ← being predicted
 MED      X    XX   XXX     X   ← being predicted
 HGH      X     X    XX   XXX   ← being predicted
```

If low/medium/high show all zeros in the confusion matrix — the model has collapsed to predicting everything as normal. Regenerate the CSV and retrain.

### 5 — Start Kafka infrastructure

```bash
docker-compose up -d
```

Wait 15 seconds, then verify all three containers are running:

```bash
docker ps
# Expected output:
# zookeeper   Up
# kafka       Up
# kafka-ui    Up
```

Kafka UI is accessible at `http://localhost:8080` to inspect topics and messages.

### 6 — Run the live pipeline

Open **4 separate terminals** from the project root:

**Terminal A — Inference consumer:**
```bash
python -m src.kafka.consumer
```
Loads `checkpoints/best_model.pt` and subscribes to `railway.sensors.raw`. Prints warm-up progress — waits until all 20 nodes have 12 readings before making predictions (~24 seconds).

**Terminal B — Live sensor simulator:**
```bash
python -m src.kafka.live_sensor_simulator --interval 2
```
Produces one JSON message per component every 2 seconds. Automatically injects fault events at scheduled times during the run.

**Terminal C — Dashboard:**
```bash
streamlit run src/dashboard/app.py
```
Open `http://localhost:8501`. Shows "Waiting for predictions..." for the first 24 seconds, then updates automatically every 5 seconds.

**Terminal D — Airflow (optional):**
```bash
pip install apache-airflow
export AIRFLOW_HOME=~/airflow
export RAILWAY_PROJECT_ROOT=$(pwd)
airflow db init
airflow standalone
```
Open `http://localhost:8080` (Airflow UI). Enable `file_watcher_dag` and `retrain_dag`.

---

## ⏱️ Timeline After Startup

| Time | Event | What you see on dashboard |
|---|---|---|
| t = 0s | Simulator starts. Consumer warming up. | "Waiting for predictions..." |
| t = 24s | All 20 nodes have 12 readings. Predictions begin. | All nodes green. Root cause panel populates. |
| t = 30s | **T05 mechanical wear injected.** Vibration begins rising. | T05 starts changing colour. |
| t = 45s | T05 risk score climbs. Propagation reaches SW3. | T05 orange. SW3 slightly elevated. |
| t = 60s | **BR2 overheating injected.** Temperature spikes. | BR2 turns orange/red. |
| t = 90s | **SW3 electrical fault injected.** Current anomaly. | SW3 red. SG2 and T06 elevated. |
| t = 120s | Cascade reaches T04, T06, SG2. Multiple nodes elevated. | Several orange nodes. Propagation grid shows spread. |
| t = 150s | **T05 escalates to CRITICAL.** | T05 red with ring. Alert fires. Root cause = T05. |

---

## 📊 Dataset Details

| Property | Value |
|---|---|
| Total records | 400,000 (20,000 steps × 20 components) |
| Component types | Tracks (10), Switches (4), Signals (3), Bridges (3) |
| Sensor channels | vibration, temperature, load, current |
| Failure events | 17 overlapping events, 6 time clusters |
| Label distribution | ~81% normal, ~8% low, ~6% medium, ~5% high |
| risk_score source | Degradation state + Gaussian noise (not sensor formula) |

### Failure event schedule

| Cluster | Steps (×20K/5K) | Origin | Failure mode | Propagates to |
|---|---|---|---|---|
| 1 — Early | 600–2,200 | T05, SW1 | mechanical_wear, electrical_fault | SW3, T04, T06, T01, T02 |
| 2 — Mid | 2,800–4,400 | BR2, T08, SG2 | overheating, mechanical_wear | T05, T06, T07, T09 |
| 3 — Heavy cascade | 5,600–7,600 | T05, BR1 | mechanical_wear, overheating | SW2, SW3, T02, T03 |
| 4 — Recovery/relapse | 8,400–10,600 | SW4, T03 | electrical_fault, mechanical_wear | T07–T09, T02, T04 |
| 5 — Third cluster | 11,600–14,200 | BR3, T01, SG3 | overheating, mechanical_wear | T08, T09, T02, SW3 |
| 6 — Major finale | 15,600–19,800 | T05, BR2, SW1 | all three modes | Wide cascade — 10+ nodes affected |

---

## 📈 Dashboard Reference

| Panel | Description |
|---|---|
| **KPI row** | Total predictions · high risk count now · medium risk count · root cause component · model accuracy · AUC-ROC |
| **Network graph** | Railway schematic with accurate node positions. Node size scales with risk score. Node colour = risk class. Edges glow red on high-risk connections. Root cause node gets a golden ring. |
| **Root cause card** | Highlighted card showing the component with highest attention weight, its risk class and score |
| **Root cause attention** | Bar chart ranking all 20 components by attention weight, colour-coded by risk class |
| **Component drill-down** | Select any component from the sidebar. Shows risk class, score, attention weight, and a 50-step risk trend line with HIGH/MEDIUM/LOW threshold bands |
| **Current risk bar chart** | All 20 components side by side. Threshold lines at 0.75 (HIGH) and 0.50 (MEDIUM) |
| **Failure propagation view** | Grid: component (rows) × last 20 prediction steps (columns). Cell colour = risk class. Shows cascade spreading visually across time. |
| **Risk heatmap** | All 20 components × last 30 predictions. Continuous green→yellow→orange→red colour scale |
| **Prediction history table** | Last 40 predictions with timestamp, root cause, high-risk components, average risk score |
| **Model performance** | Accuracy, AUC-ROC, MAE, RMSE + per-class precision/recall/F1 table |
| **Retraining history** | Populated when Airflow retrain_dag runs — shows date, trigger reason, and post-retrain metrics |

---

---

## 📂 Kafka Topics Reference

| Topic | Producer | Consumer | Message content |
|---|---|---|---|
| `railway.sensors.raw` | `live_sensor_simulator.py` | `consumer.py` | Raw JSON sensor readings per component — vibration, temperature, load, current |
| `railway.sensors.processed` | `consumer.py` | — | Feature-enriched dicts including rolling stats |
| `railway.alerts` | `consumer.py` | Airflow / monitoring system | Structured alert: component, risk level, mode, timestamp, health index |

---

## ⚙️ Airflow DAGs

Airflow is entirely optional. The core pipeline (Steps 1–6) works without it. Airflow provides automation on top.

| DAG | Trigger | Action |
|---|---|---|
| `file_watcher_dag` | Every 1 minute | Scans `data/incoming/` for new CSV files. For each file found: creates a Kafka producer, streams all rows to `railway.sensors.raw`, moves the file to `data/processed_incoming/`. |
| `retrain_dag` | Sunday 2 AM | Reads `evaluation_results.json`. If `overall.accuracy < 0.75`: runs full data regeneration + retraining + evaluation. Logs result to `data/processed/retrain_log.json`. |

```bash
pip install apache-airflow
export AIRFLOW_HOME=~/airflow
export RAILWAY_PROJECT_ROOT=$(pwd)
airflow db init
airflow standalone

# Open http://localhost:8080
# Default credentials: admin / (shown in terminal on first run)
# Enable both DAGs from the UI toggle
```

---
