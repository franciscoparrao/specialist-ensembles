# Specialist Ensembles for Cost-Sensitive Imbalanced Classification

Code and results for the manuscript:

> *Specialist Ensembles for Cost-Sensitive Imbalanced Classification: When Detecting Minority Cases Matters Most.* Francisco Parra, 2026. (Under review.)

Specialist Ensembles train sub-ensembles with deliberately different prediction biases — minority specialists, balanced specialists, and majority specialists — and learn how to combine them. The principal combiner (**SE-Stacking**) is a cost-weighted logistic meta-learner fitted on out-of-fold specialist predictions; a ratio-matched head is the recommended deployment mode.

## Key results (equal-budget tuning, 22 datasets, 13 methods, 3 seeds)

- SE-Stacking ranks **first in expected cost** at cost ratios 5 and 20 (Friedman p < 1e-10, Nemenyi CD = 3.89) with the highest mean recall (0.891).
- SE-Static is statistically tied with Self-Paced Ensemble for the best F1 rank.
- Against the trivial All-Positive classifier, ratio-matched SE-Stacking remains preferable on 16–18 of 22 datasets throughout cost ratios 10–50.

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the experiments

Datasets (18 public sources + 4 KEEL high-imbalance) are downloaded once and cached:

```bash
python phase8_datasets.py                 # builds datasets_phase8.pkl (network required once)
```

| Experiment (paper section) | Script | Output |
|---|---|---|
| Weighting-scheme ablation (§7.3) | `python phase8_weighting_schemes.py` | `phase8_weighting_results*.csv` |
| Naive + Random Balance baselines (§7.2) | `python phase8_naive_rb_baselines.py` | `phase8_naive_rb_results*.csv` |
| Main equal-budget tuned comparison, 13 methods × 3 seeds (§7.2, §7.4) | `python phase9_extended_tuning.py` | `phase9_tuned_results*.csv` |
| Ratio-matched stacking (§7.2) | `python phase9_ratio_matched.py` | `phase9_ratio_matched.csv` |
| Figures + LaTeX table rows | `python phase8_figures.py` | `figures/*.pdf` |

All experiments use fixed seeds (42/43/44) and 5-fold stratified outer CV. The exact 6-configuration tuning grids for every method are in `results/phase8_tuning_grids.csv` (also in the paper's appendix). Precomputed results for every table in the paper are included under `results/`.

## Repository layout

- `ase_implementation.py` — Adaptive Specialist Ensemble (ASE) with theoretical analysis (H, H* threshold)
- `dynamic_specialist_selection.py` — DSS (locally adaptive weights) and static specialists
- `phase7_expanded_experiments.py` — baseline implementations (SMOTEBoost, RUSBoost, EasyEnsemble, BalancedBagging, threshold-moving)
- `phase8_*.py`, `phase9_*.py` — experiment harnesses (see table above); `phase9_extended_tuning.py` also contains the simplified Self-Paced Ensemble (Liu et al., ICDE 2020)
- `results/` — CSVs backing every table in the paper
- `figures/` — publication figures

## License

MIT — see `LICENSE`.
