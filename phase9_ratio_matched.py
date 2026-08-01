"""
Phase 9: Ratio-matched stacking (review Issue 4)
================================================
The phase-8 SE-Stacking trains its logistic meta-learner with a fixed class
weight 1:5 and is then evaluated at cost ratios 1..50. This experiment tests
the natural variant: refit the meta-learner with class weights matched to the
evaluation ratio (specialists unchanged, only the logistic head is refit).

For each fold: specialists trained once, OOF matrix computed once; then one
cost-weighted logistic head per ratio in {1,2,5,10,20,50}. Reported:
- expected cost per instance at ratio r using the head matched to r
- versus the fixed rho=5 head at the same r (phase-8 scheme)
- per-dataset crossover rho* vs the All-Positive classifier for both.

Outputs: phase9_ratio_matched.csv
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from phase8_datasets import load_datasets
from phase8_weighting_schemes import (
    train_specialists, specialist_probs, oof_specialist_probs,
    eval_predictions, COST_RATIOS
)


def head(P_oof, y, ratio):
    m = LogisticRegression(class_weight={0: 1, 1: ratio}, max_iter=1000)
    m.fit(P_oof, y)
    return m


def run_dataset(data, n_splits=5, seed=42):
    X = StandardScaler().fit_transform(data['X'])
    y = data['y'].astype(int)
    rows = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        P_oof = oof_specialist_probs(X[tr], y[tr], seed)
        models, _ = train_specialists(X[tr], y[tr], seed)
        P_test = specialist_probs(models, X[te])

        fixed = head(P_oof, y[tr], 5)
        proba_fixed = fixed.predict_proba(P_test)[:, 1]
        m = eval_predictions(y[te], proba_fixed)
        m.update({'Dataset': data['name'], 'IR': data['IR'],
                  'Scheme': 'stacking_fixed5', 'fold': fold})
        rows.append(m)

        # matched: cost at ratio r evaluated with the head trained at r
        matched_costs = {}
        for r in COST_RATIOS:
            hm = head(P_oof, y[tr], r)
            pred = (hm.predict_proba(P_test)[:, 1] >= 0.5).astype(int)
            fn = int(((y[te] == 1) & (pred == 0)).sum())
            fp = int(((y[te] == 0) & (pred == 1)).sum())
            matched_costs[f'cost_r{r}'] = (r * fn + fp) / len(y[te])
            if r == 5:
                m5 = eval_predictions(y[te], hm.predict_proba(P_test)[:, 1])
        row = {'Dataset': data['name'], 'IR': data['IR'],
               'Scheme': 'stacking_matched', 'fold': fold,
               'Recall': m5['Recall'], 'Precision': m5['Precision'],
               'F1': m5['F1'], 'BAcc': m5['BAcc'], 'AUC': m5['AUC']}
        row.update(matched_costs)
        rows.append(row)
    return rows


if __name__ == "__main__":
    datasets = load_datasets()
    all_rows = Parallel(n_jobs=4, verbose=5)(
        delayed(run_dataset)(d) for d in datasets
    )
    df = pd.DataFrame([r for rows in all_rows for r in rows])
    df.to_csv('phase9_ratio_matched_folds.csv', index=False)
    agg = df.groupby(['Dataset', 'IR', 'Scheme']).mean(numeric_only=True) \
            .drop(columns=['fold']).reset_index()
    agg.to_csv('phase9_ratio_matched.csv', index=False)

    cost_cols = [f'cost_r{r}' for r in COST_RATIOS]
    print("\nMEAN EXPECTED COST PER INSTANCE")
    print(agg.groupby('Scheme')[cost_cols].mean().round(4).to_string())

    # crossover vs AllPositive (constant = majority prevalence per dataset)
    naive = pd.read_csv('phase8_naive_rb_results.csv')
    apn = naive[naive.Scheme == 'AllPositive'].set_index('Dataset')

    def linfit(row):
        y_ = np.array([row[c] for c in cost_cols], dtype=float)
        b = (y_[-1] - y_[0]) / (COST_RATIOS[-1] - COST_RATIOS[0])
        return y_[0] - b * COST_RATIOS[0], b

    print("\nCROSSOVER rho* vs AllPositive (mediana por esquema)")
    for scheme in ['stacking_fixed5', 'stacking_matched']:
        sub = agg[agg.Scheme == scheme].set_index('Dataset')
        rs = []
        for ds in sub.index:
            a_m, b_m = linfit(sub.loc[ds])
            a_ap, _ = linfit(apn.loc[ds])
            rs.append((a_ap - a_m) / b_m if b_m > 1e-9 else np.inf)
        rs = np.array(rs)
        fin = rs[np.isfinite(rs)]
        print(f"  {scheme}: mediana rho*={np.median(fin):.1f} | rho*<5 en "
              f"{int((fin<5).sum())} | 5-30: {int(((fin>=5)&(fin<=30)).sum())} | "
              f">30: {int((rs>30).sum())} (inf: {int((~np.isfinite(rs)).sum())})")
    print("\nSaved: phase9_ratio_matched.csv")
