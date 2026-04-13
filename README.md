# aisc_retention — v1.1

Python package and data for "Operator-Specific Information Retention in
Spectral Membrane Analysis" (ERI, AISC 2026 v1.1).

## New in v1.1
- `aisc_retention/operators.py` adds `decimate_edge_collapse()`: a
  topology-preserving edge-collapse decimator with manifold checks (link
  condition, face-flip rejection).
- `aisc_data/decimation_edge_collapse.csv` (30 rows): new experiment
  comparing random-subsample decimation to edge-collapse across 5
  morphologies × 6 retention levels (100%–50%).
- `figs/aisc_fig5_simplifier_comparison.png`: 2-panel comparison
  figure (LCC fraction and Δ(λ₂,μ₂) by simplifier).

## Structure
- `aisc_retention/` — Python package: utils, meshgen, descriptors, operators
- `aisc_data/` — CSV data from all experiments
- `figs/` — PNG figures (1–5)
- `run_*.py` — experiment driver scripts
- `make_figures.py` — figure generator

## Dependencies
NumPy, SciPy, Matplotlib.

## Reproducing results
```
python3 run_experiments.py       # baselines, clipping, decimation v2
python3 run_edge_collapse.py     # new in v1.1
python3 run_truncation.py        # spectral truncation + disagreement
python3 make_figures.py          # figures 1–5
```
