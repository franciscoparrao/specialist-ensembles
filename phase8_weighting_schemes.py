"""
Phase 8: Weighting scheme comparison (JMLR reviewer response)
=============================================================
Addresses rejection comments C1/C2/C3/C8:
- C1: type multipliers are heuristic and unjustified
- C2: specialist recall computed on training data instead of holdout
- C3: why minority recall as the weighting metric vs. other metrics
- C8: compare against stacking with a cost-sensitive loss (R2 suggestion)

Design: for each outer CV fold the 15 specialists (5 minority-oversampled,
5 balanced-undersampled, 5 bootstrap-majority; GB base as in the paper) are
trained ONCE. All weighting schemes are then derived from the same shared
out-of-fold (OOF) prediction matrix, so the comparison isolates the weighting
component exactly.

Schemes:
  uniform            w_i = 1/K
  type               w_i = m_type (1.5 / 1.2 / 0.8)            [paper Static]
  type_recall_train  w_i = r_i(train) * m_type                 [paper text, C2 target]
  type_recall_oof    w_i = r_i(OOF) * m_type                   [C2 fix]
  type_f1_oof        w_i = F1_i(OOF) * m_type                  [C3 ablation]
  type_gmean_oof     w_i = Gmean_i(OOF) * m_type               [C3 ablation]
  type_bacc_oof      w_i = BAcc_i(OOF) * m_type                [C3 ablation]
  bias_prop_oof      w_i ~ B_i(OOF) - min(B) + 0.1             [theory-guided, Thm 4]
  stacking_cost      logistic meta-learner on OOF probs,
                     class_weight {0:1, 1:5}                   [C8, R2 suggestion]

Outputs: phase8_weighting_results.csv (per dataset x scheme, incl. expected
cost per instance at C_FN/C_FP in {1,2,5,10,20,50}).
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, recall_score, precision_score,
                             roc_auc_score, balanced_accuracy_score)
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

from phase8_datasets import load_datasets

N_PER_TYPE = 5
TYPE_MULTIPLIERS = {'minority': 1.5, 'balanced': 1.2, 'majority': 0.8}
COST_RATIOS = [1, 2, 5, 10, 20, 50]
STACKING_COST_RATIO = 5  # the paper's target regime C_FN/C_FP >= 5


def make_base(seed):
    return GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=seed)


def build_specialist_data(X, y, spec_type, seed):
    """Resampled training set for one specialist (paper architecture)."""
    rng = np.random.RandomState(seed)
    min_mask = y == 1
    X_min, X_maj = X[min_mask], X[~min_mask]
    n_min, n_maj = len(X_min), len(X_maj)

    if spec_type == 'minority':
        X_min_over = resample(X_min, n_samples=n_maj, random_state=seed)
        Xs = np.vstack([X_min_over, X_maj])
        ys = np.concatenate([np.ones(n_maj), np.zeros(n_maj)])
    elif spec_type == 'balanced':
        X_maj_under = resample(X_maj, n_samples=n_min, replace=False, random_state=seed)
        Xs = np.vstack([X_min, X_maj_under])
        ys = np.concatenate([np.ones(n_min), np.zeros(n_min)])
    else:  # majority: bootstrap of the original distribution
        idx = rng.choice(len(y), size=len(y), replace=True)
        Xs, ys = X[idx], y[idx]

    perm = rng.permutation(len(ys))
    return Xs[perm], ys[perm]


def train_specialists(X, y, seed):
    """Train the 15 specialists; returns (models, types)."""
    models, types = [], []
    s = 0
    for spec_type in ['minority', 'balanced', 'majority']:
        for i in range(N_PER_TYPE):
            Xs, ys = build_specialist_data(X, y, spec_type, seed + s)
            clf = make_base(seed + s)
            clf.fit(Xs, ys)
            models.append(clf)
            types.append(spec_type)
            s += 1
    return models, types


def specialist_probs(models, X):
    """(n, K) matrix of minority-class probabilities."""
    return np.column_stack([m.predict_proba(X)[:, 1] for m in models])


def oof_specialist_probs(X, y, seed, n_inner=3):
    """OOF minority-class probabilities for each specialist position via
    internal stratified CV (same seeds -> same resampling recipe)."""
    P = np.full((len(y), 3 * N_PER_TYPE), np.nan)
    skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        models, _ = train_specialists(X[tr], y[tr], seed)
        P[va] = specialist_probs(models, X[va])
    return P


def per_specialist_metric(P, y, metric):
    """Metric of each specialist's thresholded prediction (cols of P)."""
    vals = []
    for k in range(P.shape[1]):
        pred = (P[:, k] >= 0.5).astype(int)
        if metric == 'recall':
            v = recall_score(y, pred, zero_division=0)
        elif metric == 'f1':
            v = f1_score(y, pred, zero_division=0)
        elif metric == 'gmean':
            rec = recall_score(y, pred, zero_division=0)
            spec = recall_score(1 - y, 1 - pred, zero_division=0)
            v = np.sqrt(rec * spec)
        elif metric == 'bacc':
            v = balanced_accuracy_score(y, pred)
        vals.append(v)
    return np.array(vals)


def normalize(w):
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 1e-6, None)
    return w / w.sum()


def compute_weights(P_train, P_oof, y_train, types):
    """All linear weighting schemes from shared information."""
    mult = np.array([TYPE_MULTIPLIERS[t] for t in types])
    schemes = {}

    schemes['uniform'] = normalize(np.ones(len(types)))
    schemes['type'] = normalize(mult)

    r_train = per_specialist_metric(P_train, y_train, 'recall')
    schemes['type_recall_train'] = normalize(r_train * mult)

    for metric in ['recall', 'f1', 'gmean', 'bacc']:
        r_oof = per_specialist_metric(P_oof, y_train, metric)
        name = 'type_recall_oof' if metric == 'recall' else f'type_{metric}_oof'
        schemes[name] = normalize(r_oof * mult)

    # Theory-guided bias-proportional weights (Theorem 4), from OOF
    min_mask = y_train == 1
    biases = P_oof[min_mask].mean(axis=0) - 0.5
    schemes['bias_prop_oof'] = normalize(biases - biases.min() + 0.1)

    return schemes


def expected_costs(y_true, pred):
    """Expected cost per instance for each C_FN/C_FP ratio (C_FP = 1)."""
    fn = int(((y_true == 1) & (pred == 0)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    n = len(y_true)
    return {f'cost_r{r}': (r * fn + fp) / n for r in COST_RATIOS}


def eval_predictions(y_true, proba, prefix_extra=None):
    pred = (proba >= 0.5).astype(int)
    out = {
        'Recall': recall_score(y_true, pred, zero_division=0),
        'Precision': precision_score(y_true, pred, zero_division=0),
        'F1': f1_score(y_true, pred, zero_division=0),
        'BAcc': balanced_accuracy_score(y_true, pred),
        'AUC': roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else np.nan,
    }
    out.update(expected_costs(y_true, pred))
    return out


def run_dataset(data, n_splits=5, seed=42):
    X = StandardScaler().fit_transform(data['X'])
    y = data['y'].astype(int)

    rows = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        X_tr, y_tr, X_te, y_te = X[tr], y[tr], X[te], y[te]

        P_oof = oof_specialist_probs(X_tr, y_tr, seed)
        models, types = train_specialists(X_tr, y_tr, seed)
        P_train = specialist_probs(models, X_tr)
        P_test = specialist_probs(models, X_te)

        weights = compute_weights(P_train, P_oof, y_tr, types)
        for scheme, w in weights.items():
            proba = P_test @ w
            m = eval_predictions(y_te, proba)
            m.update({'Dataset': data['name'], 'IR': data['IR'],
                      'Scheme': scheme, 'fold': fold})
            rows.append(m)

        # C8: cost-weighted logistic stacking on OOF probs
        meta = LogisticRegression(class_weight={0: 1, 1: STACKING_COST_RATIO},
                                  max_iter=1000)
        meta.fit(P_oof, y_tr)
        proba = meta.predict_proba(P_test)[:, 1]
        m = eval_predictions(y_te, proba)
        m.update({'Dataset': data['name'], 'IR': data['IR'],
                  'Scheme': 'stacking_cost', 'fold': fold})
        rows.append(m)

    return rows


if __name__ == "__main__":
    datasets = load_datasets()
    print(f"Datasets: {len(datasets)}")

    all_rows = Parallel(n_jobs=6, verbose=10)(
        delayed(run_dataset)(d) for d in datasets
    )
    df = pd.DataFrame([r for rows in all_rows for r in rows])
    df.to_csv('phase8_weighting_results_folds.csv', index=False)

    # Aggregate over folds
    agg = df.groupby(['Dataset', 'IR', 'Scheme']).mean(numeric_only=True) \
            .drop(columns=['fold']).reset_index()
    agg.to_csv('phase8_weighting_results.csv', index=False)

    print("\n" + "=" * 70)
    print("MEAN OVER DATASETS (per scheme)")
    print("=" * 70)
    cols = ['Recall', 'Precision', 'F1', 'BAcc', 'AUC', 'cost_r5', 'cost_r20']
    print(agg.groupby('Scheme')[cols].mean().round(4).to_string())

    print("\nMEAN RANK (Recall, lower=better) / (cost_r5, lower cost rank=better)")
    for metric, ascending in [('Recall', False), ('cost_r5', True), ('F1', False)]:
        pivot = agg.pivot_table(index='Dataset', columns='Scheme', values=metric)
        ranks = pivot.rank(axis=1, ascending=ascending).mean().sort_values()
        print(f"\n-- {metric} --")
        print(ranks.round(2).to_string())

    print("\nSaved: phase8_weighting_results.csv / phase8_weighting_results_folds.csv")
