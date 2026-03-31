# Soccer Analytics Pipeline

A progressive series of models for valuing player and team actions in soccer, built on StatsBomb open data.

---

## Models

### 1. VAEP — Valuing Actions by Estimating Probabilities
**Files:** `ingest.py` → `features.py` → `train.py`

Event-based value model. Converts StatsBomb events to SPADL format, engineers rolling 3-action window features, and trains two XGBoost classifiers to estimate P(scores) and P(concedes) within the next 10 actions.

```
VAEP(a) = [P_scores(after) − P_scores(before)] − [P_concedes(after) − P_concedes(before)]
```

**Data:** Bayer Leverkusen 2023/24 Bundesliga season (34 matches)

---

### 2. Spatial xG — GNN Shot Value Model
**Files:** `freeze_frames.py` → `gnn_model.py` → `train_gnn.py`

Replaces location-only xG with a Graph Attention Network that reads the full player geometry at the moment of each shot. Nodes = players, edges = teammate/proximity connections. Clusters high-value shot configurations to identify the spatial patterns that generate goals.

**Data:** StatsBomb shot freeze frames (908 shots, Bundesliga 2023/24)

---

### 3. PCVA — Possession Chain Value Attribution
**File:** `chain_value.py`

Fixes VAEP's blindness to downstream contribution. Groups actions into possession chains, computes the terminal value of each chain (did it produce a shot?), and distributes that value back to every player in the chain using exponential decay weighting.

---

### 4. xVAEP — Expected VAEP
**File:** `xvaep.py`

Separates decision quality from outcome luck. Trains an action-success model to estimate P(success | action, context), then computes counterfactual VAEP under both success and failure outcomes. xVAEP = P(success) × VAEP_if_success + P(failure) × VAEP_if_failure.

Corrects VAEP's systematic undervaluation of creative, high-difficulty players — Florian Wirtz moves from 23rd/23 (VAEP) to 9th/23 (xVAEP).

---

### 5. STVN — Spatial-Temporal Value Network
**Files:** `ingest_360.py` → `build_chains.py` → `stvn.py` → `train_stvn.py`

Deep learning value model. Each possession chain is encoded as a sequence of spatial graphs — one per event — where node features include position, velocity, and acceleration derived from consecutive freeze frames. A shared GATConv frame encoder produces per-frame embeddings; a GRU integrates them across the chain; an MLP head predicts P(chain produces a shot).

```
[G_1, G_2, ..., G_T]  →  FrameEncoder (GATConv × 2)  →  GRU  →  P(shot)
```

The GRU hidden state at each timestep h_t represents accumulated chain value. Δh_t = marginal value added by action t — a spatiotemporal analogue of xVAEP.

**Data:** FIFA World Cup 2022 (64 matches) + UEFA Euro 2024 (51 matches) + Bundesliga 2023/24 (34 matches) — 149 matches, ~66k possession chains.

**Train on GPU:** see `colab_train.ipynb`

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, XGBoost requires OpenMP:
```bash
brew install libomp
```

---

## Running the pipelines

### VAEP
```bash
python ingest.py       # fetch Bundesliga data → data/raw/
python features.py     # SPADL conversion + feature engineering → data/features/
python train.py        # train XGBoost, compute VAEP values → data/outputs/
```

### Spatial xG
```bash
python freeze_frames.py   # parse shot freeze frames → data/graphs.pkl
python train_gnn.py       # train GAT, cluster game states → data/outputs/
python plot_top_shots.py  # visualise top 20% shots → data/outputs/top_shots_p*.png
```

### PCVA + xVAEP
```bash
python chain_value.py  # possession chain attribution
python xvaep.py        # expected VAEP with counterfactuals
```

### STVN (run on GPU via Colab)
```bash
python ingest_360.py   # fetch 360 data for WC2022 + Euro2024 + Bundesliga
python build_chains.py # build possession chain graph sequences → data/360/chains.pkl
python train_stvn.py   # train STVN → models/stvn.pt
```

---

## Data

All data is fetched from [StatsBomb Open Data](https://github.com/statsbomb/open-data) via `statsbombpy`. No data files are committed to this repo.

| Dataset | Matches | 360 Coverage |
|---|---|---|
| Bundesliga 2023/24 | 34 | 83% of events |
| FIFA World Cup 2022 | 64 | 82% of events |
| UEFA Euro 2024 | 51 | 83% of events |

---

## Key findings

- **Granit Xhaka** led Leverkusen in total VAEP (+21.3) across 6,395 actions
- **Florian Wirtz** ranked last in VAEP (−17.2) but **1st in shot-chain involvement** (41.3% of all shot-producing chains) — VAEP systematically penalises creative, high-difficulty players
- **xVAEP** (AUC 0.937 on action-success model) corrects this: Wirtz jumps from 23rd → 9th
- **Spatial xG clusters** reveal that the defining geometry of high-value shots is not just distance — it's the number of defenders in the shooting lane (cluster 3: 2.0 opponents between shooter and goal vs 4.7 average)
