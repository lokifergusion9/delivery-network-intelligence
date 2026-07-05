# 🚚 Delivery Network Intelligence

Graph-based ETA prediction and bottleneck analysis for a logistics delivery network. Instead of treating each delivery leg as an independent route, this project models the network as a directed graph — facilities as nodes, corridors as edges — and uses graph structure (centrality, connectivity) to dramatically improve ETA accuracy over a naive routing estimate (OSRM), while also pinpointing exactly which hubs are driving network-wide delay.

### 🌐 [Live Demo](https://delivery-network-intelligence-bgrkkktakv7jqgvm7msawv.streamlit.app/)

> Hosted on Streamlit Community Cloud's free tier — the app sleeps after a period of inactivity, so the first load after a while may take 30–60 seconds to wake up.

## The problem

The routing engine (OSRM) assumes clean roads and shortest paths. Real operations don't work that way: across the analyzed trips, the typical leg takes about **2x longer than OSRM predicts**, and roughly 96% of corridors run more than 20% slower than promised. That breaks SLAs, hurts customer trust, and undermines capacity planning built on those estimates.

## What this does

1. **Builds a corridor graph** from raw shipment scan data — source/destination facilities as nodes, aggregated corridor stats (volume, delay ratio, SLA-breach contribution) as edges.
2. **Computes structural features** per facility: betweenness centrality, in/out degree, clustering coefficient.
3. **Scores delay risk per hub** (0–100) by blending realized SLA-breach volume with structural centrality.
4. **Trains a graph-enhanced LightGBM ETA model** (base route features + facility graph features) and compares it head-to-head against OSRM and a graph-free ML baseline.
5. **Surfaces actionable results**: bottleneck hub rankings, corridor-level drill-downs, and an interactive ETA predictor.

### Results (held-out test set)

| ETA method | Within 15% of actual | MAE |
|---|---|---|
| OSRM (naive) | ~5% of legs | 107 |
| ML baseline (no graph) | ~35% of legs | 42 |
| **Graph-enhanced model** | **~48% of legs** | **32** |

Graph features alone cut average error by ~24% and nearly triple the share of legs predicted within 15% of reality, versus OSRM.

See [`strategy_memo.md`](strategy_memo.md) for the full write-up, including the top 5 bottleneck hubs, recommended interventions, and quantified business impact.

## Repo contents

| File | Description |
|---|---|
| `delhivery_graph_eta.ipynb` | Main analysis notebook: data prep, graph construction, centrality/risk scoring, node2vec embeddings, model training and evaluation. |
| `app.py` | Interactive Streamlit dashboard driven by the same pipeline as the notebook. |
| `delivery_data.csv` | Raw shipment/segment-level delivery data (Delhivery). |
| `strategy_memo.md` | Business-facing summary of findings and recommendations. |
| `requirements.txt` | Python dependencies. |

## Getting started

### Requirements

- Python 3.9+

### Installation

```bash
git clone https://github.com/lokifergusion9/delivery-network-intelligence.git
cd delivery-network-intelligence
pip install -r requirements.txt
```

### Run the notebook

```bash
jupyter notebook delhivery_graph_eta.ipynb
```

### Run the dashboard

```bash
streamlit run app.py
```

The dashboard includes:
- **🗺️ Network Map** — corridor graph visualization, node size/color = delay-risk score
- **🏭 Hub Risk** — facilities ranked by delay-risk score, with per-hub corridor drill-down
- **🛣️ Corridor Risk** — corridors ranked by SLA-breach contribution
- **🔮 ETA Predictor** — predict actual leg time for a chosen source/destination and compare against the naive OSRM estimate
- **📊 Model** — baseline vs. graph-enhanced model comparison on held-out data

## How it works (pipeline summary)

1. Raw segment-level scans are aggregated into origin-destination (OD) legs per trip.
2. OD legs are aggregated further into corridors (source → destination pairs) with trip volume, median delay ratio, and an SLA-breach score (`trips × max(median_delay_ratio − 1, 0)`).
3. A directed graph is built from corridors, weighted by delay ratio.
4. Betweenness centrality, degree, and clustering are computed per facility node.
5. A 0–100 risk score blends realized breach (70%) with structural centrality (30%).
6. A LightGBM regressor is trained twice — once on base route features only, once with graph features added — to isolate the lift from network structure.

## Notes

- `delivery_data.csv` is tracked in this repo; if you'd rather not version raw data, uncomment the relevant line in `.gitignore`.
- The full node2vec embedding variant of the model lives in the notebook; `app.py` uses the lighter base + centrality feature set for fast, cache-friendly predictions in the dashboard.
