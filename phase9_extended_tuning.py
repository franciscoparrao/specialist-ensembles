"""
Phase 9: Extended equal-budget comparison (simulated-review response)
=====================================================================
Extends the phase 8 tuned comparison to address the pre-submission review:
- Issue 1: DSS included under the SAME equal-budget protocol as every method.
- Issue 2: a post-2020 competitor added — Self-Paced Ensemble (SPE; Liu et
  al., ICDE 2020), simplified implementation in the style of the paper's
  other baselines (GB base for comparability).
- Minor (single seed): the whole comparison runs over seeds {42, 43, 44};
  per-dataset scores are averaged over seeds before Friedman ranking.

13 methods x 22 datasets x 5 outer folds x 3 seeds, budget 6 configs each.

Outputs: phase9_tuned_results.csv (+ _folds.csv), same layout as phase 8.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

from phase8_datasets import load_datasets
from phase8_weighting_schemes import eval_predictions
from phase8_uniform_tuning import GRIDS as GRIDS8
from dynamic_specialist_selection import DynamicSpecialistEnsemble


class SelfPacedEnsemble(BaseEstimator, ClassifierMixin):
    """Self-Paced Ensemble (Liu et al., ICDE 2020), simplified.

    Iteratively trains base classifiers on the minority class plus a
    self-paced undersample of the majority class: majority hardness w.r.t.
    the current ensemble is split into k bins, and bins are sampled with
    weight 1/(h_bin + alpha_i), where the self-paced factor alpha_i grows
    with the iteration (easy-first, then uniform). GB base (50, depth 3)
    for comparability with the other ensembles in the study.
    """

    def __init__(self, n_estimators=20, k_bins=10, random_state=None):
        self.n_estimators = n_estimators
        self.k_bins = k_bins
        self.random_state = random_state

    def _base(self, i):
        return GradientBoostingClassifier(n_estimators=50, max_depth=3,
                                          random_state=(self.random_state or 0) + i)

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.classes_ = np.unique(y)
        X_min, X_maj = X[y == 1], X[y == 0]
        n_min, n_maj = len(X_min), len(X_maj)
        self.estimators_ = []

        # f_0: balanced random undersample
        idx = rng.choice(n_maj, size=min(n_min, n_maj), replace=False)
        Xs = np.vstack([X_min, X_maj[idx]])
        ys = np.concatenate([np.ones(n_min), np.zeros(len(idx))])
        clf = self._base(0)
        clf.fit(Xs, ys)
        self.estimators_.append(clf)

        for i in range(1, self.n_estimators):
            proba_maj = np.mean([c.predict_proba(X_maj)[:, 1]
                                 for c in self.estimators_], axis=0)
            hardness = proba_maj  # majority true label is 0: hardness = P(1|x)
            bins = np.minimum((hardness * self.k_bins).astype(int), self.k_bins - 1)
            alpha = np.tan(np.pi / 2 * i / max(self.n_estimators - 1, 1) * 0.99)

            weights = np.zeros(n_maj)
            for b in range(self.k_bins):
                mask = bins == b
                if mask.sum() == 0:
                    continue
                h_b = hardness[mask].mean()
                weights[mask] = 1.0 / (h_b + alpha + 1e-9) / mask.sum()
            if weights.sum() <= 0:
                weights = np.ones(n_maj)
            weights = weights / weights.sum()

            idx = rng.choice(n_maj, size=min(n_min, n_maj), replace=False, p=weights)
            Xs = np.vstack([X_min, X_maj[idx]])
            ys = np.concatenate([np.ones(n_min), np.zeros(len(idx))])
            clf = self._base(i)
            clf.fit(Xs, ys)
            self.estimators_.append(clf)
        return self

    def predict_proba(self, X):
        p = np.mean([c.predict_proba(X)[:, 1] for c in self.estimators_], axis=0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


GRIDS = dict(GRIDS8)
GRIDS['SPE'] = [SelfPacedEnsemble(n, k, random_state=42)
                for n in [10, 20, 30] for k in [5, 10]]
GRIDS['DSS'] = [DynamicSpecialistEnsemble(n_specialists=n, adaptation_strength=a,
                                          random_state=42)
                for n in [9, 15] for a in [0.5, 1.0, 1.5]]


def tune_and_eval(X_tr, y_tr, X_te, y_te, configs, seed):
    X_sub, X_val, y_sub, y_val = train_test_split(
        X_tr, y_tr, test_size=0.33, stratify=y_tr, random_state=seed)
    scores = []
    for cfg in configs:
        try:
            c = clone(cfg)
            if hasattr(c, 'random_state'):
                c.random_state = seed
            c.fit(X_sub, y_sub)
            scores.append(f1_score(y_val, c.predict(X_val), zero_division=0))
        except Exception:
            scores.append(0.0)
    best_idx = int(np.argmax(scores))
    best = clone(configs[best_idx])
    if hasattr(best, 'random_state'):
        best.random_state = seed
    best.fit(X_tr, y_tr)
    return best.predict_proba(X_te)[:, 1], best_idx


def run_unit(data, seed, fold, tr, te):
    X = StandardScaler().fit_transform(data['X'])
    y = data['y'].astype(int)
    rows = []
    for name, configs in GRIDS.items():
        try:
            proba, best_idx = tune_and_eval(X[tr], y[tr], X[te], y[te], configs, seed)
            m = eval_predictions(y[te], proba)
            m.update({'Dataset': data['name'], 'IR': data['IR'], 'Method': name,
                      'seed': seed, 'fold': fold, 'best_config': best_idx})
            rows.append(m)
        except Exception as e:
            print(f"  {data['name']}/{name}/s{seed}f{fold}: ERROR {str(e)[:60]}")
    return rows


if __name__ == "__main__":
    datasets = load_datasets()
    SEEDS = [42, 43, 44]
    units = []
    for seed in SEEDS:
        for d in sorted(datasets, key=lambda d: -(d['n_samples'] * d['n_features'])):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for fold, (tr, te) in enumerate(skf.split(d['X'], d['y'].astype(int))):
                units.append((d, seed, fold, tr, te))
    print(f"Methods: {len(GRIDS)} | Units: {len(units)} (datasets x folds x seeds)")

    all_rows = Parallel(n_jobs=10, verbose=10)(
        delayed(run_unit)(d, s, f, tr, te) for d, s, f, tr, te in units
    )
    df = pd.DataFrame([r for rows in all_rows for r in rows])
    df.to_csv('phase9_tuned_results_folds.csv', index=False)

    agg = df.groupby(['Dataset', 'IR', 'Method']).mean(numeric_only=True) \
            .drop(columns=['fold', 'seed', 'best_config']).reset_index()
    agg.to_csv('phase9_tuned_results.csv', index=False)

    cols = ['Recall', 'Precision', 'F1', 'AUC', 'cost_r5', 'cost_r20']
    print(agg.groupby('Method')[cols].mean().round(4).to_string())
    for metric, asc in [('Recall', False), ('F1', False), ('cost_r5', True)]:
        ranks = agg.pivot_table(index='Dataset', columns='Method', values=metric) \
                   .rank(axis=1, ascending=asc).mean().sort_values()
        print(f"\n-- rank {metric} --\n{ranks.round(2).to_string()}")
    print("\nSaved: phase9_tuned_results.csv / _folds.csv")
