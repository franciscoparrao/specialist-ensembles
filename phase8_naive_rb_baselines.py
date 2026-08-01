"""
Phase 8: Naive + Random Balance baselines (JMLR reviewer response)
==================================================================
Addresses rejection comments C7/C9:
- C7 (R1): "comparisons should also be made against the naive procedure of
  predicting all positive class" in the high C_FN/C_FP regime.
- C9 (R2): missing literature/baseline — Diez-Pastor et al. (2015),
  "Random Balance: Ensembles of variable priors classifiers for imbalanced
  data", Knowledge-Based Systems 85.

Baselines:
  AllPositive     predict 1 for everything (recall=1, precision=prevalence)
  AllNegative     predict 0 for everything (reference lower bound)
  RandomBalance   15 members; each trained on a same-size dataset whose class
                  proportion is drawn uniformly at random; minority inflated
                  with SMOTE, majority randomly undersampled (Diez-Pastor
                  2015, simplified in the style of the paper's SMOTEBoost).
                  GB base (50, depth 3) for comparability with the Static
                  specialists.

Also re-evaluates the paper's Static specialists and the phase8 stacking
variant on the same folds so the cost curves are directly comparable.

Outputs: phase8_naive_rb_results.csv (+ _folds.csv), same metric layout as
phase8_weighting_schemes.py (expected cost at C_FN/C_FP in {1,2,5,10,20,50}).
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from phase8_datasets import load_datasets
from phase8_weighting_schemes import (
    train_specialists, specialist_probs, oof_specialist_probs,
    compute_weights, eval_predictions, STACKING_COST_RATIO
)


class AllPositive(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        return np.column_stack([np.zeros(len(X)), np.ones(len(X))])

    def predict(self, X):
        return np.ones(len(X), dtype=int)


class AllNegative(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X)), np.zeros(len(X))])

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def smote_sample(X_min, n_samples, k_neighbors=5, rng=None):
    """Synthetic minority samples (same simplified SMOTE as phase7)."""
    rng = rng or np.random
    if len(X_min) < 2:
        return X_min[rng.randint(0, len(X_min), size=n_samples)] if len(X_min) else X_min
    k = min(k_neighbors, len(X_min) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
    synthetic = []
    for _ in range(n_samples):
        i = rng.randint(len(X_min))
        _, neigh = nn.kneighbors([X_min[i]])
        j = neigh[0, rng.randint(1, k + 1)]
        gap = rng.random()
        synthetic.append(X_min[i] + gap * (X_min[j] - X_min[i]))
    return np.array(synthetic)


class RandomBalance(BaseEstimator, ClassifierMixin):
    """Random Balance ensemble (Diez-Pastor et al., 2015), simplified.

    Each member is trained on a dataset of the original size n whose minority
    proportion p is drawn from U(2/n, 1 - 2/n): the minority class is grown
    with SMOTE when round(p*n) exceeds its size and the majority class is
    randomly undersampled to fill the rest (and vice versa).
    """

    def __init__(self, n_estimators=15, random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.classes_ = np.unique(y)
        X_min, X_maj = X[y == 1], X[y == 0]
        n = len(y)

        self.estimators_ = []
        for i in range(self.n_estimators):
            p = rng.uniform(2 / n, 1 - 2 / n)
            n_min_target = max(2, int(round(p * n)))
            n_maj_target = max(2, n - n_min_target)

            def sized(X_cls, target):
                if target <= len(X_cls):
                    idx = rng.choice(len(X_cls), size=target, replace=False)
                    return X_cls[idx]
                extra = smote_sample(X_cls, target - len(X_cls), rng=rng)
                return np.vstack([X_cls, extra]) if len(extra) else X_cls

            X_min_s = sized(X_min, n_min_target)
            X_maj_s = sized(X_maj, n_maj_target)
            Xs = np.vstack([X_min_s, X_maj_s])
            ys = np.concatenate([np.ones(len(X_min_s)), np.zeros(len(X_maj_s))])
            perm = rng.permutation(len(ys))

            clf = GradientBoostingClassifier(n_estimators=50, max_depth=3,
                                             random_state=(self.random_state or 0) + i)
            clf.fit(Xs[perm], ys[perm])
            self.estimators_.append(clf)
        return self

    def predict_proba(self, X):
        p = np.mean([c.predict_proba(X)[:, 1] for c in self.estimators_], axis=0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def run_dataset(data, n_splits=5, seed=42):
    X = StandardScaler().fit_transform(data['X'])
    y = data['y'].astype(int)

    rows = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        X_tr, y_tr, X_te, y_te = X[tr], y[tr], X[te], y[te]

        # Naive + Random Balance
        for name, clf in [('AllPositive', AllPositive()),
                          ('AllNegative', AllNegative()),
                          ('RandomBalance', RandomBalance(random_state=seed))]:
            clf.fit(X_tr, y_tr)
            proba = clf.predict_proba(X_te)[:, 1]
            m = eval_predictions(y_te, proba)
            m.update({'Dataset': data['name'], 'IR': data['IR'],
                      'Scheme': name, 'fold': fold})
            rows.append(m)

        # Reference: Static (type weights) and stacking on the same folds
        P_oof = oof_specialist_probs(X_tr, y_tr, seed)
        models, types = train_specialists(X_tr, y_tr, seed)
        P_train = specialist_probs(models, X_tr)
        P_test = specialist_probs(models, X_te)
        weights = compute_weights(P_train, P_oof, y_tr, types)

        for scheme in ['type', 'bias_prop_oof']:
            proba = P_test @ weights[scheme]
            m = eval_predictions(y_te, proba)
            m.update({'Dataset': data['name'], 'IR': data['IR'],
                      'Scheme': f'ASE_{scheme}', 'fold': fold})
            rows.append(m)

        meta = LogisticRegression(class_weight={0: 1, 1: STACKING_COST_RATIO},
                                  max_iter=1000)
        meta.fit(P_oof, y_tr)
        proba = meta.predict_proba(P_test)[:, 1]
        m = eval_predictions(y_te, proba)
        m.update({'Dataset': data['name'], 'IR': data['IR'],
                  'Scheme': 'ASE_stacking_cost', 'fold': fold})
        rows.append(m)

    return rows


if __name__ == "__main__":
    datasets = load_datasets()
    print(f"Datasets: {len(datasets)}")

    all_rows = Parallel(n_jobs=6, verbose=10)(
        delayed(run_dataset)(d) for d in datasets
    )
    df = pd.DataFrame([r for rows in all_rows for r in rows])
    df.to_csv('phase8_naive_rb_results_folds.csv', index=False)

    agg = df.groupby(['Dataset', 'IR', 'Scheme']).mean(numeric_only=True) \
            .drop(columns=['fold']).reset_index()
    agg.to_csv('phase8_naive_rb_results.csv', index=False)

    print("\n" + "=" * 70)
    print("MEAN OVER DATASETS")
    print("=" * 70)
    cost_cols = [c for c in agg.columns if c.startswith('cost_')]
    cols = ['Recall', 'Precision', 'F1'] + cost_cols
    print(agg.groupby('Scheme')[cols].mean().round(4).to_string())

    print("\nCROSSOVER ANALYSIS: mean expected cost per instance by cost ratio")
    print("(where AllPositive beats every trained method = upper bound of applicability)")
    by_scheme = agg.groupby('Scheme')[cost_cols].mean()
    print(by_scheme.round(4).to_string())

    print("\nSaved: phase8_naive_rb_results.csv / _folds.csv")
