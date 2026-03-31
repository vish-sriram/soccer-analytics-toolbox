# Soccer Analytics — Spatiotemporal Player Valuation

A progressive series of five models for valuing player actions in soccer, built on StatsBomb open data. Each model addresses a limitation of the previous one, culminating in a spatiotemporal deep learning value network that reads full player geometry and chain history to estimate expected reward.

---

## Model Progression

```
StatsBomb Events ──▶  vaep/       Classical XGBoost action valuation
                   ──▶  pcva/      Possession chain attribution
                   ──▶  xvaep/     Expected VAEP (counterfactual correction)

StatsBomb Shots  ──▶  spatial_xg/ Graph Attention Network spatial xG

StatsBomb 360    ──▶  stvn/        Spatiotemporal Value Network (GATConv + GRU)
```

---

## Models

### `vaep/` — VAEP: Valuing Actions by Estimating Probabilities

Two XGBoost classifiers estimate the probability that a team scores or concedes within the next 10 actions. Per-action value is the change in those probabilities caused by each action.

```
VAEP(a) = [P_scores(after) − P_scores(before)] − [P_concedes(after) − P_concedes(before)]
```

- **CV AUC:** 0.724 (scores model), 0.739 (concedes model)
- **Data:** Bayer Leverkusen 2023/24 Bundesliga (34 matches, 80k actions)

---

### `spatial_xg/` — Spatial xG: Graph Attention Network

Replaces location-only xG with a full player geometry model. At the moment of each shot, visible players become graph nodes; edges connect teammates (fully) and nearby opponents (proximity-based). A two-layer GAT encodes the graph; global pooling feeds an MLP head predicting P(goal).

- **CV AUC:** 0.727
- **Key finding:** K-means clustering over graph embeddings identifies 6 shot geometries. Top cluster (33% goal rate) is defined by **≤2 defenders in the shooting lane** — not distance alone.
- **Data:** 908 shot freeze frames, Bundesliga 2023/24

---

### `pcva/` — PCVA: Possession Chain Value Attribution

VAEP is blind to downstream contribution: a player who draws defenders and creates space gets no credit if their action "fails." PCVA groups actions into possession chains, assigns the chain's terminal value (did it produce a shot?) and distributes that value back to every player in the chain using exponential decay weighting.

- **Key finding:** Florian Wirtz appears in **41.3% of all Leverkusen shot-producing chains** — more than any player including Xhaka — yet ranked last in VAEP.

---

### `xvaep/` — xVAEP: Expected VAEP

Separates decision quality from outcome luck, analogous to xG vs goals. Trains an action-success model (AUC 0.937) to estimate P(success | action type, location, context), then computes counterfactual VAEP under both success and failure outcomes.

```
xVAEP(a) = P(success) × VAEP_if_success  +  P(failure) × VAEP_if_failure
```

- **Key finding:** Wirtz jumps from **23rd/23 → 9th/23**. The largest correction on the squad. Defenders (Tah +18, Tapsoba +10) also surge — both are systematically penalised by VAEP for high-difficulty defensive actions.

---

### `stvn/` — STVN: Spatial-Temporal Value Network

A deep learning value model trained on possession chains from StatsBomb 360 data, where every event has a full player freeze frame. Each chain becomes a variable-length sequence of spatial graphs. Node features include position, velocity, and acceleration derived from consecutive frames.

```
Chain: [G₁, G₂, ..., Gₜ]

  ┌─────────────────────────────┐
  │  Frame Encoder (shared)     │  GATConv × 2  →  global pool  →  e_t
  └──────────────┬──────────────┘
                 ↓  sequence [e₁ … eₜ]
  ┌─────────────────────────────┐
  │  Temporal Encoder           │  2-layer GRU  →  h_t per step
  └──────────────┬──────────────┘
                 ↓  h_T
              MLP head  →  P(chain produces a shot)
```

The GRU hidden state h_t represents accumulated chain value at each step. **Δh_t = marginal value of action t** — a spatiotemporal analogue of xVAEP sensitive to player movement and chain history.

- **Training data:** FIFA World Cup 2022 + UEFA Euro 2024 + Bundesliga 2023/24 — 149 matches, 65,789 possession chains
- **GPU training:** `notebooks/colab_train.ipynb` (T4 ~30 min)

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

macOS only — XGBoost requires OpenMP:
```bash
brew install libomp
```

---

## Running the Pipelines

All commands run from the **project root**.

```bash
# VAEP
python vaep/ingest.py
python vaep/features.py
python vaep/train.py

# Spatial xG
python spatial_xg/ingest.py
python spatial_xg/train.py
python spatial_xg/visualise.py   # saves data/outputs/top_shots_p*.png

# PCVA + xVAEP
python pcva/chain_value.py
python xvaep/xvaep.py

# STVN — run on GPU via Colab (see notebooks/colab_train.ipynb)
python stvn/ingest.py
python stvn/build_chains.py
python stvn/train.py
```

---

## Repository Layout

```
├── vaep/               VAEP XGBoost pipeline
│   ├── ingest.py       Fetch StatsBomb events
│   ├── features.py     SPADL conversion + feature engineering
│   └── train.py        Train models, compute per-action VAEP
│
├── spatial_xg/         Spatial xG GNN pipeline
│   ├── ingest.py       Parse shot freeze frames → PyG graphs
│   ├── model.py        GATConv model definition
│   ├── train.py        Train, cross-validate, cluster game states
│   └── visualise.py    Plot top-20% shots by P(goal)
│
├── pcva/               Possession Chain Value Attribution
│   └── chain_value.py  Chain construction + decay-weighted attribution
│
├── xvaep/              Expected VAEP
│   └── xvaep.py        Action-success model + counterfactual VAEP
│
├── stvn/               Spatial-Temporal Value Network
│   ├── ingest.py       Fetch 360 event data (WC2022, Euro2024, Bundesliga)
│   ├── build_chains.py Possession chains with velocity/acceleration features
│   ├── model.py        STVN architecture (GATConv + GRU)
│   └── train.py        Train + team-level value attribution
│
├── notebooks/
│   └── colab_train.ipynb  GPU training on Google Colab
│
├── requirements.txt
└── .gitignore
```

---

## Data

All data fetched from [StatsBomb Open Data](https://github.com/statsbomb/open-data) — no files committed to this repo.

| Dataset | Matches | 360 Coverage |
|---|---|---|
| Bundesliga 2023/24 (Bayer Leverkusen) | 34 | 83% of events |
| FIFA World Cup 2022 | 64 | 82% of events |
| UEFA Euro 2024 | 51 | 83% of events |
