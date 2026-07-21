# Adaptive PSO for Demand Forecasting

Comparing a standard Particle Swarm Optimisation (PSO) algorithm against a custom, adaptive variant, applied to a demand-prediction (regression) problem. Built as part of my dissertation/coursework project.

## Problem

`DemandPrediction` fits a linear model (bias + 13 weighted demand indicators) to training data, using **Mean Absolute Error (MAE)** as the cost function. Instead of solving this with gradient descent, the weights are treated as a 14-dimensional search space and optimised using PSO — each "particle" is a candidate set of weights, and the swarm collectively searches for the combination that minimises MAE.

## What's in this repo

| File | Description |
|---|---|
| `Baseline_PSO.py` | Standard PSO implementation — fixed inertia weight, no velocity limits |
| `Novel_Variant_PSO.py` | My custom variant (see below) |
| `DemandPrediction.py` | The optimisation problem: loads data, defines bounds, evaluates candidate solutions |
| `Main.py` | Simple random-search baseline, included for reference/sanity-checking |

## The novel variant — what's different and why

The baseline PSO uses a fixed inertia weight and unbounded velocity, which can cause particles to overshoot good regions of the search space or converge too early. My variant adds three changes aimed at balancing exploration and exploitation more effectively:

1. **Adaptive inertia weight** — starts high (more exploration) and linearly decreases over the run (more exploitation as the swarm converges), rather than staying fixed.
2. **Velocity clamping** — caps each particle's velocity as a proportion of its search-space bounds, preventing particles from overshooting promising regions.
3. **Stagnation detection and reset** — tracks how many iterations each particle has gone without improving its personal best. If a particle stagnates past a threshold, its velocity is re-randomised to reintroduce diversity into the swarm, rather than letting it get permanently stuck.

## Results

Both variants train on `data/train.csv` and are evaluated on `data/test.csv`, using 50 particles over 1000 iterations. Results from each run are logged to `baseline_results.csv` / `novel_results.csv`.

| | Train MAE | Test MAE |
|---|---|---|
| Baseline PSO | _add your value_ | _add your value_ |
| Novel Variant PSO | _add your value_ | _add your value_ |

*(Run both scripts and fill in your actual numbers here — a results table is one of the first things people look at.)*

## Setup

```bash
pip install -r requirements.txt
```

Place your training and test data at `data/train.csv` and `data/test.csv` (no header row, first column = target value, remaining columns = demand indicators).

## Running

```bash
python Baseline_PSO.py
python Novel_Variant_PSO.py
```

Each run prints progress every 10 iterations and appends final train/test MAE to its results CSV.

## Known issue

There's a known indexing bug in `DemandPrediction.py`'s prediction function (uses a fixed index instead of iterating over all indicators) — flagged here for transparency, fix in progress.

## Possible extensions

- Hyperparameter tuning for `w_max`, `w_min`, `stagnation_limit`
- Compare against other metaheuristics (e.g. genetic algorithms, differential evolution)
- Cross-validation instead of a single train/test split
