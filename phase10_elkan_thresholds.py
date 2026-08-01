"""
Phase 10: Elkan-threshold baselines (blind-review Issue B2)
===========================================================
Under expected-cost evaluation, the decision-theoretically optimal rule for a
calibrated probabilistic classifier is the Elkan (2001) threshold
tau* = C_FP / (C_FP + C_FN) = 1/(1+rho). This experiment gives that rule to
the strongest probabilistic baselines and to the raw specialist pool, so the
value of the specialist POOL is separated from the value of a cost-aware
DECISION RULE:

  GB+Elkan, RF+Elkan, SPE+Elkan, RandomBalance+Elkan   (no specialist pool)
  Pool-uniform+Elkan                                    (pool, trivial combiner)
  vs. stacking_matched (phase9_ratio_matched.csv)       (pool + learned head)

Each model is fit ONCE per fold; cost at ratio r uses threshold 1/(1+r)
("+Elkan") or 0.5 (reference). Wall-clock fit times are recorded per method.

Same folds as phase9_ratio_matched (seed 42, 5-fold stratified).

Outputs: phase10_elkan_results.csv (+ _folds.csv), stdout summary.
"""

import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from phase8_datasets import load_datasets
from phase8_weighting_schemes import (train_specialists, specialist_probs,
                                      COST_RATIOS)
from phase8_naive_rb_baselines import RandomBalance
from phase9_extended_tuning import SelfPacedEnsemble


def cost_row(y_true, proba, thresholds):
    """Expected cost per instance at each ratio with the given threshold rule."""
    out = {}
    for r in COST_RATIOS:
        tau = thresholds(r)
        pred = (proba >= tau).astype(int)
        fn = int(((y_true == 1) & (pred == 0)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        out[f'cost_r{r}'] = (r * fn + fp) / len(y_true)
    return out


def run_dataset(data, n_splits=5, seed=42):
    X = StandardScaler().fit_transform(data['X'])
    y = data['y'].astype(int)
    rows = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        X_tr, y_tr, X_te, y_te = X[tr], y[tr], X[te], y[te]

        models = {
            'GradientBoost': GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                        random_state=seed),
            'RandomForest': RandomForestClassifier(n_estimators=200,
                                                   class_weight='balanced',
                                                   random_state=seed),
            'SPE': SelfPacedEnsemble(n_estimators=20, k_bins=10, random_state=seed),
            'RandomBalance': RandomBalance(n_estimators=15, random_state=seed),
        }
        probas, fit_times = {}, {}
        for name, clf in models.items():
            t0 = time.time()
            clf.fit(X_tr, y_tr)
            fit_times[name] = time.time() - t0
            probas[name] = clf.predict_proba(X_te)[:, 1]

        # Raw specialist pool, uniform combiner (no learned head)
        t0 = time.time()
        pool, _ = train_specialists(X_tr, y_tr, seed)
        fit_times['Pool-uniform'] = time.time() - t0
        probas['Pool-uniform'] = specialist_probs(pool, X_te).mean(axis=1)

        for name, proba in probas.items():
            for rule, thr in [('Elkan', lambda r: 1.0 / (1.0 + r)),
                              ('t05', lambda r: 0.5)]:
                m = cost_row(y_te, proba, thr)
                m.update({'Dataset': data['name'], 'IR': data['IR'],
                          'Method': f'{name}+{rule}', 'fold': fold,
                          'fit_seconds': fit_times[name]})
                rows.append(m)
    return rows


if __name__ == "__main__":
    datasets = load_datasets()
    all_rows = Parallel(n_jobs=6, verbose=5)(
        delayed(run_dataset)(d) for d in datasets
    )
    df = pd.DataFrame([r for rows in all_rows for r in rows])
    df.to_csv('phase10_elkan_results_folds.csv', index=False)
    agg = df.groupby(['Dataset', 'IR', 'Method']).mean(numeric_only=True) \
            .drop(columns=['fold']).reset_index()
    agg.to_csv('phase10_elkan_results.csv', index=False)

    cost_cols = [f'cost_r{r}' for r in COST_RATIOS]
    print("\nMEAN EXPECTED COST PER INSTANCE (Elkan threshold vs 0.5)")
    print(agg.groupby('Method')[cost_cols].mean().round(4).to_string())

    rm = pd.read_csv('phase9_ratio_matched.csv')
    mt = rm[rm.Scheme == 'stacking_matched'].groupby('Scheme')[cost_cols].mean()
    print("\nReferencia (mismos folds): SE-Stacking ratio-matched")
    print(mt.round(4).to_string())

    print("\nWALL-CLOCK: mean fit seconds per fold")
    ft = df[df.Method.str.endswith('+Elkan')].copy()
    ft['Method'] = ft.Method.str.replace('+Elkan', '', regex=False)
    print(ft.groupby('Method')['fit_seconds'].mean().round(2).to_string())
