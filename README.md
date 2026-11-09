# ✈️ AirRoute Intelligence

**Temporal Link Prediction on the Global Flight Network**

A complete Graph Neural Network project applying Temporal Graph Networks (TGN) to predict future flight connectivity, using the [tgbl-flight](https://tgb.complexdatalab.com/) dataset from the Temporal Graph Benchmark (NeurIPS 2023).

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-red)
![PyG](https://img.shields.io/badge/PyG-2.8-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.48-ff4b4b)

---

## Overview

Given the history of flight interactions between airports, the task is to predict which airports will be connected at a future time step. The project follows the official TGB evaluation protocol with the One-vs-Many strategy (1 positive + 20 negatives per edge).

### Key Results (Test Set)

| Model | Type | MRR | Hits@10 |
|-------|------|-----|---------|
| **HistFreq** | Heuristic | **0.738** | 0.934 |
| Recency | Heuristic | 0.704 | 0.931 |
| **TGN** | Neural (GNN) | 0.653 | **0.938** |
| Random | Heuristic | 0.174 | 0.476 |
| EdgeBank | Heuristic | 0.163 | 0.935 |

**Main finding**: The global flight network is extremely regular — historical frequency alone achieves the best MRR. TGN excels at identifying plausible destinations (best Hits@10) but its fine-grained ranking is less precise than simple frequency counting.

---

## Dataset

**tgbl-flight** from the Temporal Graph Benchmark:

| Property | Value |
|----------|-------|
| Airports | 18,143 |
| Flight interactions | 67,169,570 |
| Period | January 2019 – October 2022 |
| Resolution | Daily |
| Split | Train 70% / Val 15% / Test 15% (temporal) |

---

## Project Structure

```
├── app.py                          # Streamlit dashboard (5 pages)
├── requirements.txt
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Phase 1: Dataset exploration
│   ├── 02_temporal_graph_analysis.ipynb  # Phase 2: Graph construction
│   ├── 03_baselines.ipynb          # Phase 3: Baseline models
│   ├── 04_tgn_model.ipynb          # Phase 4: TGN training
│   ├── 05_evaluation.ipynb         # Phase 5: Test evaluation
│   └── 06_resilience.ipynb         # Phase 6: Network resilience
├── src/
│   ├── data/
│   │   ├── preprocessing.py        # Chunked CSV-to-numpy pipeline
│   │   └── loader.py               # Data loading utilities
│   ├── baselines/
│   │   ├── baselines.py            # Random, HistFreq, Recency, EdgeBank
│   │   └── evaluate.py             # Evaluation harness
│   ├── models/
│   │   ├── tgn.py                  # TGN architecture (memory + link predictor)
│   │   └── train.py                # Training loop with batched evaluation
│   └── graph/
│       ├── temporal_graph.py       # TemporalFlightGraph wrapper
│       └── resilience.py           # Centrality, communities, disruption simulation
├── scripts/
│   ├── download_dataset.py         # Download tgbl-flight from TGB
│   ├── run_baselines.py            # Run all baselines
│   ├── run_test_evaluation.py      # Full test set evaluation
│   ├── prepare_dashboard_data.py   # Precompute dashboard data (airport coords, predictions)
│   └── plot_*.py                   # Figure generation scripts
└── results/
    ├── figures/                    # 17 generated figures
    ├── baseline_results.json       # Validation results
    ├── test_results.json           # Test results (all models)
    ├── tgn_results.json            # TGN configuration and results
    ├── tgn_training_log.json       # Epoch-by-epoch training log
    └── resilience_results.json     # Network resilience metrics
```

---

## Installation

```bash
# Clone
git clone https://github.com/ranimnsr/AirRoute-Intelligence.git
cd AirRoute-Intelligence

# Install dependencies
pip install -r requirements.txt

# Download dataset (~2.5 GB)
python scripts/download_dataset.py

# Preprocess (creates data/processed/)
python -c "from src.data.preprocessing import preprocess_tgbl_flight; preprocess_tgbl_flight()"
```

---

## Usage

### Run the Streamlit Dashboard

```bash
# Precompute dashboard data (airport coordinates, predictions)
python scripts/prepare_dashboard_data.py

# Launch
streamlit run app.py
```

The dashboard has 5 interactive pages:

| Page | Description |
|------|-------------|
| **Overview** | World map with 14,500+ routes, KPIs, key results |
| **Airport Explorer** | Select any airport — connections, centrality, interactive map |
| **Future Link Predictions** | Select source airport, view predicted destinations with scores |
| **Network Resilience Simulator** | Simulate airport removal, measure connectivity impact |
| **Model Performance** | Compare all models with interactive Plotly charts |

### Train the TGN Model

```bash
python -m src.models.train
```

Default configuration (CPU-optimized): 3 months training data, 2 epochs, batch size 200, memory dim 100.

### Run Baselines

```bash
python scripts/run_baselines.py
```

### Full Test Evaluation

```bash
python scripts/run_test_evaluation.py
```

---

## Model Architecture

The TGN (Temporal Graph Network) model consists of:

| Component | Role |
|-----------|------|
| **TGNMemory** | Per-node memory vector updated at each interaction |
| **IdentityMessage** | Message = concatenation of source and destination memories |
| **LastAggregator** | Keeps the most recent message per node |
| **LinkPredictor** | 3-layer MLP: [z_src ‖ z_dst] → probability score |

Total parameters: 201,701. Trained with BPR loss (softplus(neg - pos)).

---

## Network Resilience Analysis

Graph analytics (not GNN) on a 3-month flight network snapshot:

- **13,986 airports**, **505,101 routes**, **4.8M flights**
- **KORD (Chicago O'Hare)**: #1 by degree and PageRank
- **PANC (Anchorage)**: #1 by betweenness — trans-Pacific bridge
- **67 communities** detected (Louvain algorithm)
- After removing the **50 most critical airports**: connectivity drops only from 85.6% to 84.1%
- The network is **remarkably resilient** thanks to its distributed hub-and-spoke structure

---

## Hardware

This project was developed and run entirely on CPU:

- Windows 11, Python 3.12.6
- PyTorch 2.10.0 (CPU), PyG 2.8.1
- ~2.3 GB free RAM
- Training: ~27 min (2 epochs, 3 months data)
- Test evaluation: ~56 min (with memory warmup)

---

## References

- Huang et al., *Temporal Graph Benchmark for Machine Learning on Temporal Graphs*, NeurIPS 2023
- Rossi et al., *Temporal Graph Networks for Deep Learning on Dynamic Graphs*, ICML 2020 Workshop
- tgbl-flight dataset: https://tgb.complexdatalab.com/

---

## License

This project is for educational purposes (engineering school project).
