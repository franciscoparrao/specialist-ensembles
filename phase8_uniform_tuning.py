"""
Phase 8: Uniform equal-budget tuning of all methods (JMLR reviewer response)
============================================================================
Addresses rejection comment C5 (R1): "Insufficient detail is given towards
how the competing algorithms are tuned. For example, for the boosting +
optimal threshold algorithm, there are no details on how the boosting
procedure is tuned."

Protocol (documented for the paper's appendix):
- Every method receives the SAME budget: 6 hyperparameter configurations.
- Selection: inner stratified validation split (33% of the training
  partition), maximizing F1 (neutral w.r.t. the recall-vs-precision
  trade-off).
- The winning configuration is refit on the full training partition and
  evaluated on the outer test fold (5-fold outer stratified CV).
- Thresholds (for +Threshold methods) are optimized inside fit() as before,
  i.e. they are part of the method, not of the tuning budget.

Methods tuned (11): RF+Threshold, GB+Threshold, EasyEnsemble, RUSBoost,
SMOTEBoost, BalancedBagging, RF(balanced), GradientBoost, RandomBalance,
ASE-Static(type weights), ASE-Stacking(cost-weighted logistic).

Outputs: phase8_tuned_results.csv (+ _folds.csv), phase8_tuning_grids.csv
(the exact grids, for the appendix table).
"""

import numpy as np
import pandas as pd
from itertools import product
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

from phase8_datasets import load_datasets
from phase8_weighting_schemes import (
    N_PER_TYPE, TYPE_MULTIPLIERS, make_base, build_specialist_data,
    specialist_probs, eval_predictions, STACKING_COST_RATIO
)
from phase8_naive_rb_baselines import RandomBalance
from phase7_expanded_experiments import (
    ThresholdOptimizer, EasyEnsemble, RUSBoost, SMOTEBoost, BalancedBagging
)


# --------------------------------------------------------------------------
# Parametrized ASE (n_per_type, gb_estimators tunable; both weight schemes)
# --------------------------------------------------------------------------

class ASETunable(BaseEstimator, ClassifierMixin):
    def __init__(self, n_per_type=5, gb_estimators=50, scheme='type', random_state=42):
        self.n_per_type = n_per_type
        self.gb_estimators = gb_estimators
        self.scheme = scheme  # 'type' | 'stacking'
        self.random_state = random_state

    def _train(self, X, y):
        models, types = [], []
        s = 0
        for spec_type in ['minority', 'balanced', 'majority']:
            for _ in range(self.n_per_type):
                Xs, ys = build_specialist_data(X, y, spec_type, self.random_state + s)
                clf = GradientBoostingClassifier(
                    n_estimators=self.gb_estimators, max_depth=3,
                    random_state=self.random_state + s)
                clf.fit(Xs, ys)
                models.append(clf)
                types.append(spec_type)
                s += 1
        return models, types

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if self.scheme == 'stacking':
            # OOF probs for the meta-learner
            P = np.full((len(y), 3 * self.n_per_type), np.nan)
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            for tr, va in skf.split(X, y):
                models, _ = self._train(X[tr], y[tr])
                P[va] = specialist_probs(models, X[va])
            self.meta_ = LogisticRegression(
                class_weight={0: 1, 1: STACKING_COST_RATIO}, max_iter=1000)
            self.meta_.fit(P, y)
        self.models_, self.types_ = self._train(X, y)
        if self.scheme == 'type':
            w = np.array([TYPE_MULTIPLIERS[t] for t in self.types_])
            self.weights_ = w / w.sum()
        return self

    def predict_proba(self, X):
        P = specialist_probs(self.models_, X)
        if self.scheme == 'stacking':
            p = self.meta_.predict_proba(P)[:, 1]
        else:
            p = P @ self.weights_
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# --------------------------------------------------------------------------
# Grids: exactly 6 configurations per method (equal budget)
# --------------------------------------------------------------------------

def rf_thresh(n, d):
    return ThresholdOptimizer(RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42))

def gb_thresh(n, d):
    return ThresholdOptimizer(GradientBoostingClassifier(n_estimators=n, max_depth=d, random_state=42))

class EasyEnsembleT(EasyEnsemble):
    """EasyEnsemble with tunable AdaBoost size."""
    def __init__(self, n_subsets=10, ada_estimators=30, random_state=None):
        super().__init__(n_subsets=n_subsets, random_state=random_state)
        self.ada_estimators = ada_estimators

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.classes_ = np.unique(y)
        minority = y == 1
        X_min, X_maj = X[minority], X[~minority]
        n_min = len(X_min)
        self.estimators_ = []
        for i in range(self.n_subsets):
            idx = np.random.choice(len(X_maj), size=min(n_min, len(X_maj)), replace=False)
            X_sub = np.vstack([X_min, X_maj[idx]])
            y_sub = np.array([1] * n_min + [0] * len(idx))
            clf = AdaBoostClassifier(n_estimators=self.ada_estimators, random_state=self.random_state)
            clf.fit(X_sub, y_sub)
            self.estimators_.append(clf)
        return self


GRIDS = {
    'RF+Threshold': [rf_thresh(n, d) for n, d in product([100, 200], [None, 10, 20])],
    'GB+Threshold': [gb_thresh(n, d) for n, d in product([50, 100, 200], [2, 3])],
    'EasyEnsemble': [EasyEnsembleT(s, a, random_state=42)
                     for s, a in product([5, 10, 15], [30, 60])],
    'RUSBoost': [RUSBoost(n_estimators=n, random_state=42) for n in [20, 30, 50, 75, 100, 150]],
    'SMOTEBoost': [SMOTEBoost(n_estimators=n, k_neighbors=k, random_state=42)
                   for n, k in product([30, 50, 100], [3, 5])],
    'BalancedBagging': [BalancedBagging(n_estimators=n, random_state=42)
                        for n in [5, 10, 15, 20, 30, 50]],
    'RandomForest': [RandomForestClassifier(n_estimators=n, max_depth=d,
                                            class_weight='balanced', random_state=42)
                     for n, d in product([100, 200, 300], [None, 15])],
    'GradientBoost': [GradientBoostingClassifier(n_estimators=n, max_depth=d, random_state=42)
                      for n, d in product([50, 100, 200], [2, 3])],
    'RandomBalance': [RandomBalance(n_estimators=n, random_state=42)
                      for n in [5, 10, 15, 20, 25, 35]],
    'ASE-Static': [ASETunable(p, g, scheme='type') for p, g in product([3, 5, 8], [50, 100])],
    'ASE-Stacking': [ASETunable(p, g, scheme='stacking') for p, g in product([3, 5, 8], [50, 100])],
}


def describe_config(clf):
    return repr(clf)[:120]


def tune_and_eval(X_tr, y_tr, X_te, y_te, configs, seed=42):
    """Select by F1 on a stratified validation split, refit best, evaluate."""
    from sklearn.model_selection import train_test_split
    X_sub, X_val, y_sub, y_val = train_test_split(
        X_tr, y_tr, test_size=0.33, stratify=y_tr, random_state=seed)
    scores = []
    for cfg in configs:
        try:
            c = clone(cfg)
            c.fit(X_sub, y_sub)
            scores.append(f1_score(y_val, c.predict(X_val), zero_division=0))
        except Exception:
            scores.append(0.0)
    best_idx = int(np.argmax(scores))
    best = clone(configs[best_idx])
    best.fit(X_tr, y_tr)
    proba = best.predict_proba(X_te)[:, 1]
    return proba, best_idx


def run_fold(data, fold, tr, te, seed=42):
    """One (dataset, fold) unit — parallelization grain for load balance."""
    X = StandardScaler().fit_transform(data['X'])
    y = data['y'].astype(int)
    rows = []
    for name, configs in GRIDS.items():
        try:
            proba, best_idx = tune_and_eval(X[tr], y[tr], X[te], y[te], configs, seed)
            m = eval_predictions(y[te], proba)
            m.update({'Dataset': data['name'], 'IR': data['IR'],
                      'Method': name, 'fold': fold, 'best_config': best_idx})
            rows.append(m)
        except Exception as e:
            print(f"  {data['name']}/{name}/f{fold}: ERROR {str(e)[:60]}")
    return rows


if __name__ == "__main__":
    datasets = load_datasets()
    print(f"Datasets: {len(datasets)} | Methods: {len(GRIDS)} | Budget: 6 configs c/u")

    # Persist the exact grids for the appendix
    grid_rows = [{'Method': name, 'config_idx': i, 'config': describe_config(c)}
                 for name, cfgs in GRIDS.items() for i, c in enumerate(cfgs)]
    pd.DataFrame(grid_rows).to_csv('phase8_tuning_grids.csv', index=False)

    # Build (dataset, fold) work units, largest datasets first (tail control)
    units = []
    for d in sorted(datasets, key=lambda d: -(d['n_samples'] * d['n_features'])):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (tr, te) in enumerate(skf.split(d['X'], d['y'].astype(int))):
            units.append((d, fold, tr, te))
    print(f"Work units (dataset x fold): {len(units)}")

    all_rows = Parallel(n_jobs=10, verbose=10)(
        delayed(run_fold)(d, fold, tr, te) for d, fold, tr, te in units
    )
    df = pd.DataFrame([r for rows in all_rows for r in rows])
    df.to_csv('phase8_tuned_results_folds.csv', index=False)

    agg = df.groupby(['Dataset', 'IR', 'Method']).mean(numeric_only=True) \
            .drop(columns=['fold', 'best_config']).reset_index()
    agg.to_csv('phase8_tuned_results.csv', index=False)

    print("\n" + "=" * 70)
    print("TUNED COMPARISON — MEAN OVER DATASETS")
    print("=" * 70)
    cols = ['Recall', 'Precision', 'F1', 'AUC', 'cost_r5', 'cost_r20']
    print(agg.groupby('Method')[cols].mean().round(4).to_string())

    print("\nMEAN RANKS")
    for metric, ascending in [('F1', False), ('Recall', False), ('cost_r5', True)]:
        pivot = agg.pivot_table(index='Dataset', columns='Method', values=metric)
        ranks = pivot.rank(axis=1, ascending=ascending).mean().sort_values()
        print(f"\n-- {metric} --")
        print(ranks.round(2).to_string())

    print("\nSaved: phase8_tuned_results.csv / _folds.csv / phase8_tuning_grids.csv")
