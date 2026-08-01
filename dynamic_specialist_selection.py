"""
Dynamic Specialist Selection (DSS)
==================================
Algorithmic innovation for JMLR paper.

Instead of fixed weights for all test instances, DSS selects specialists
dynamically based on local characteristics of each test instance:
1. Local minority density - are we near minority instances?
2. Uncertainty - are specialists disagreeing?
3. Distance to decision boundary

This connects to mixture-of-experts literature (gating function).
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample
from sklearn.metrics import recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')


class DynamicSpecialistEnsemble(BaseEstimator, ClassifierMixin):
    """
    Specialist Ensemble with Dynamic Selection (DSS).

    Key innovation: Instead of fixed weights, compute instance-specific
    weights based on local data characteristics.

    For each test instance x:
    1. Estimate local minority density using k-NN
    2. Compute specialist disagreement (uncertainty)
    3. Adapt weights: favor minority specialists in minority-dense regions
    """

    def __init__(self, n_specialists=15, k_neighbors=10,
                 adaptation_strength=1.0, random_state=None):
        self.n_specialists = n_specialists
        self.k_neighbors = k_neighbors
        self.adaptation_strength = adaptation_strength
        self.random_state = random_state

    def fit(self, X, y):
        np.random.seed(self.random_state)

        self.classes_ = np.unique(y)
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()

        # Identify classes
        class_counts = np.bincount(y.astype(int))
        self.minority_class_ = np.argmin(class_counts)
        self.majority_class_ = np.argmax(class_counts)

        minority_mask = y == self.minority_class_
        X_min, y_min = X[minority_mask], y[minority_mask]
        X_maj, y_maj = X[~minority_mask], y[~minority_mask]

        n_minority = len(y_min)
        n_majority = len(y_maj)

        # Build k-NN index for local density estimation
        self.nn_ = NearestNeighbors(n_neighbors=min(self.k_neighbors, len(y)))
        self.nn_.fit(X)

        self.specialists_ = []
        self.specialist_types_ = []
        self.base_weights_ = []

        # Distribute specialists
        n_minority_spec = self.n_specialists // 3
        n_balanced_spec = self.n_specialists // 3
        n_majority_spec = self.n_specialists - n_minority_spec - n_balanced_spec

        # 1. MINORITY SPECIALISTS
        for i in range(n_minority_spec):
            X_min_over = resample(X_min, n_samples=n_majority,
                                  random_state=self.random_state + i if self.random_state else None)
            y_min_over = np.full(n_majority, self.minority_class_)

            X_train = np.vstack([X_min_over, X_maj])
            y_train = np.concatenate([y_min_over, y_maj])

            idx = np.random.permutation(len(y_train))
            X_train, y_train = X_train[idx], y_train[idx]

            clf = GradientBoostingClassifier(
                n_estimators=50, max_depth=3,
                random_state=self.random_state + i if self.random_state else None
            )
            clf.fit(X_train, y_train)
            self.specialists_.append(clf)
            self.specialist_types_.append('minority')
            self.base_weights_.append(1.5)

        # 2. BALANCED SPECIALISTS
        for i in range(n_balanced_spec):
            X_maj_under = resample(X_maj, n_samples=n_minority,
                                   random_state=self.random_state + 100 + i if self.random_state else None)
            y_maj_under = np.full(n_minority, self.majority_class_)

            X_train = np.vstack([X_min, X_maj_under])
            y_train = np.concatenate([y_min, y_maj_under])

            idx = np.random.permutation(len(y_train))
            X_train, y_train = X_train[idx], y_train[idx]

            clf = GradientBoostingClassifier(
                n_estimators=50, max_depth=3,
                random_state=self.random_state + 100 + i if self.random_state else None
            )
            clf.fit(X_train, y_train)
            self.specialists_.append(clf)
            self.specialist_types_.append('balanced')
            self.base_weights_.append(1.2)

        # 3. MAJORITY SPECIALISTS
        for i in range(n_majority_spec):
            idx = np.random.choice(len(y), size=len(y), replace=True)
            X_train, y_train = X[idx], y[idx]

            clf = GradientBoostingClassifier(
                n_estimators=50, max_depth=3,
                random_state=self.random_state + 200 + i if self.random_state else None
            )
            clf.fit(X_train, y_train)
            self.specialists_.append(clf)
            self.specialist_types_.append('majority')
            self.base_weights_.append(0.8)

        # Normalize base weights
        total = sum(self.base_weights_)
        self.base_weights_ = [w / total for w in self.base_weights_]

        return self

    def _compute_local_minority_density(self, X):
        """Estimate local minority class density for each instance."""
        distances, indices = self.nn_.kneighbors(X)
        neighbor_labels = self.y_train_[indices]

        # Fraction of neighbors that are minority class
        minority_density = (neighbor_labels == self.minority_class_).mean(axis=1)
        return minority_density

    def _compute_specialist_predictions(self, X):
        """Get predictions from all specialists."""
        predictions = np.column_stack([
            clf.predict_proba(X)[:, 1] for clf in self.specialists_
        ])
        return predictions

    def _compute_uncertainty(self, predictions):
        """Compute prediction uncertainty (disagreement among specialists)."""
        # Standard deviation across specialists
        uncertainty = predictions.std(axis=1)
        return uncertainty

    def _compute_dynamic_weights(self, X):
        """
        Compute instance-specific weights for each specialist.

        Strategy:
        - High local minority density → favor minority specialists
        - High uncertainty → favor balanced specialists
        - Low minority density, low uncertainty → favor majority specialists
        """
        n_samples = len(X)
        n_specialists = len(self.specialists_)

        # Get local characteristics
        minority_density = self._compute_local_minority_density(X)
        predictions = self._compute_specialist_predictions(X)
        uncertainty = self._compute_uncertainty(predictions)

        # Normalize to [0, 1]
        if minority_density.max() > minority_density.min():
            minority_density_norm = (minority_density - minority_density.min()) / (minority_density.max() - minority_density.min())
        else:
            minority_density_norm = np.zeros_like(minority_density)

        if uncertainty.max() > uncertainty.min():
            uncertainty_norm = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
        else:
            uncertainty_norm = np.zeros_like(uncertainty)

        # Initialize weights matrix (n_samples x n_specialists)
        weights = np.zeros((n_samples, n_specialists))

        for i in range(n_samples):
            for j, spec_type in enumerate(self.specialist_types_):
                base_w = self.base_weights_[j]

                # Adjust based on local characteristics
                if spec_type == 'minority':
                    # Favor when local minority density is high
                    adjustment = 1 + self.adaptation_strength * minority_density_norm[i]
                elif spec_type == 'balanced':
                    # Favor when uncertainty is high
                    adjustment = 1 + self.adaptation_strength * uncertainty_norm[i]
                else:  # majority
                    # Favor when minority density is low and uncertainty is low
                    adjustment = 1 + self.adaptation_strength * (1 - minority_density_norm[i]) * (1 - uncertainty_norm[i])

                weights[i, j] = base_w * adjustment

            # Normalize row
            weights[i] /= weights[i].sum()

        return weights, predictions

    def predict_proba(self, X):
        """Prediction with dynamic specialist selection."""
        weights, predictions = self._compute_dynamic_weights(X)

        # Weighted combination (instance-specific)
        # predictions: (n_samples, n_specialists)
        # weights: (n_samples, n_specialists)
        weighted_preds = (weights * predictions).sum(axis=1)

        return np.column_stack([1 - weighted_preds, weighted_preds])

    def predict(self, X):
        probas = self.predict_proba(X)
        return (probas[:, 1] >= 0.5).astype(int)


class StaticSpecialistEnsemble(BaseEstimator, ClassifierMixin):
    """
    Original SpecialistEns with static weights (for comparison).
    """

    def __init__(self, n_specialists=15, random_state=None):
        self.n_specialists = n_specialists
        self.random_state = random_state

    def fit(self, X, y):
        np.random.seed(self.random_state)

        self.classes_ = np.unique(y)

        class_counts = np.bincount(y.astype(int))
        self.minority_class_ = np.argmin(class_counts)
        self.majority_class_ = np.argmax(class_counts)

        minority_mask = y == self.minority_class_
        X_min, y_min = X[minority_mask], y[minority_mask]
        X_maj, y_maj = X[~minority_mask], y[~minority_mask]

        n_minority = len(y_min)
        n_majority = len(y_maj)

        self.specialists_ = []
        self.weights_ = []

        n_minority_spec = self.n_specialists // 3
        n_balanced_spec = self.n_specialists // 3
        n_majority_spec = self.n_specialists - n_minority_spec - n_balanced_spec

        # Minority specialists
        for i in range(n_minority_spec):
            X_min_over = resample(X_min, n_samples=n_majority,
                                  random_state=self.random_state + i if self.random_state else None)
            y_min_over = np.full(n_majority, self.minority_class_)
            X_train = np.vstack([X_min_over, X_maj])
            y_train = np.concatenate([y_min_over, y_maj])
            idx = np.random.permutation(len(y_train))

            clf = GradientBoostingClassifier(n_estimators=50, max_depth=3,
                                             random_state=self.random_state + i if self.random_state else None)
            clf.fit(X_train[idx], y_train[idx])
            self.specialists_.append(clf)
            self.weights_.append(1.5)

        # Balanced specialists
        for i in range(n_balanced_spec):
            X_maj_under = resample(X_maj, n_samples=n_minority,
                                   random_state=self.random_state + 100 + i if self.random_state else None)
            y_maj_under = np.full(n_minority, self.majority_class_)
            X_train = np.vstack([X_min, X_maj_under])
            y_train = np.concatenate([y_min, y_maj_under])
            idx = np.random.permutation(len(y_train))

            clf = GradientBoostingClassifier(n_estimators=50, max_depth=3,
                                             random_state=self.random_state + 100 + i if self.random_state else None)
            clf.fit(X_train[idx], y_train[idx])
            self.specialists_.append(clf)
            self.weights_.append(1.2)

        # Majority specialists
        for i in range(n_majority_spec):
            idx = np.random.choice(len(y), size=len(y), replace=True)
            clf = GradientBoostingClassifier(n_estimators=50, max_depth=3,
                                             random_state=self.random_state + 200 + i if self.random_state else None)
            clf.fit(X[idx], y[idx])
            self.specialists_.append(clf)
            self.weights_.append(0.8)

        # Normalize
        total = sum(self.weights_)
        self.weights_ = [w / total for w in self.weights_]

        return self

    def predict_proba(self, X):
        probas = np.zeros((len(X), 2))
        for clf, w in zip(self.specialists_, self.weights_):
            probas += w * clf.predict_proba(X)
        return probas

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ============================================================
# COMPARISON EXPERIMENT
# ============================================================

if __name__ == "__main__":
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_classification
    import pandas as pd

    print("=" * 70)
    print("DYNAMIC SPECIALIST SELECTION vs STATIC WEIGHTS")
    print("=" * 70)

    def create_dataset(n_samples=2000, ir=10, random_state=42):
        n_minority = n_samples // (ir + 1)
        X, y = make_classification(
            n_samples=n_samples, n_features=20, n_informative=10,
            n_redundant=5, n_clusters_per_class=2,
            weights=[1 - n_minority/n_samples, n_minority/n_samples],
            flip_y=0.01, random_state=random_state
        )
        return X, y

    results = []

    for ir in [5, 10, 20]:
        print(f"\n{'='*50}")
        print(f"Imbalance Ratio: {ir}:1")
        print("=" * 50)

        X, y = create_dataset(n_samples=2000, ir=ir, random_state=42)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        methods = {
            'DSS (Dynamic)': DynamicSpecialistEnsemble(n_specialists=15, random_state=42),
            'Static': StaticSpecialistEnsemble(n_specialists=15, random_state=42),
        }

        for name, clf_template in methods.items():
            f1_scores = []
            recall_scores = []

            for train_idx, test_idx in skf.split(X, y):
                from sklearn.base import clone
                clf = clone(clf_template)
                clf.fit(X[train_idx], y[train_idx])

                y_pred = clf.predict(X[test_idx])
                f1_scores.append(f1_score(y[test_idx], y_pred))
                recall_scores.append(recall_score(y[test_idx], y_pred))

            results.append({
                'IR': ir,
                'Method': name,
                'F1': np.mean(f1_scores),
                'F1_std': np.std(f1_scores),
                'Recall': np.mean(recall_scores),
                'Recall_std': np.std(recall_scores)
            })

            print(f"  {name:20}: F1={np.mean(f1_scores):.3f}±{np.std(f1_scores):.3f}  "
                  f"Recall={np.mean(recall_scores):.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    df = pd.DataFrame(results)
    pivot_f1 = df.pivot_table(index='IR', columns='Method', values='F1')
    pivot_recall = df.pivot_table(index='IR', columns='Method', values='Recall')

    print("\nF1 Scores:")
    print(pivot_f1.round(3))

    print("\nRecall Scores:")
    print(pivot_recall.round(3))

    # Check if DSS improves
    dss_f1 = df[df['Method'] == 'DSS (Dynamic)']['F1'].mean()
    static_f1 = df[df['Method'] == 'Static']['F1'].mean()
    dss_recall = df[df['Method'] == 'DSS (Dynamic)']['Recall'].mean()
    static_recall = df[df['Method'] == 'Static']['Recall'].mean()

    print(f"\nDSS vs Static:")
    print(f"  F1: {dss_f1:.3f} vs {static_f1:.3f} (Δ = {dss_f1-static_f1:+.3f})")
    print(f"  Recall: {dss_recall:.3f} vs {static_recall:.3f} (Δ = {dss_recall-static_recall:+.3f})")

    df.to_csv('dss_comparison_results.csv', index=False)
    print("\nResults saved to dss_comparison_results.csv")
