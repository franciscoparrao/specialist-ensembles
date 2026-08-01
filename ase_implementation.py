"""
Adaptive Specialist Ensemble (ASE)
==================================

Implementation of the theory-guided adaptive specialist ensemble algorithm.
This module provides:
- ASE: Adaptive configuration search
- TheoreticalAnalysis: H* threshold computation
- Comparison utilities for fixed vs adaptive

Author: [Paper Authors]
License: MIT
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from scipy.stats import norm
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class SpecialistConfig:
    """Configuration for a single specialist type."""
    ratio: float  # majority:minority ratio (1.0 = balanced, inf = all majority)
    n_classifiers: int = 3

    def __repr__(self):
        if self.ratio == float('inf'):
            return f"Specialist(1:inf, n={self.n_classifiers})"
        return f"Specialist(1:{self.ratio:.1f}, n={self.n_classifiers})"


@dataclass
class EnsembleConfig:
    """Full ensemble configuration."""
    specialists: List[SpecialistConfig]

    @property
    def K(self) -> int:
        return len(self.specialists)

    @property
    def ratios(self) -> List[float]:
        return [s.ratio for s in self.specialists]


class TheoreticalAnalysis:
    """
    Compute theoretical quantities: H, H*, and predicted improvement.
    Based on Theorem 4 and Corollary 1 from the paper.
    """

    @staticmethod
    def compute_bias(predictions: np.ndarray, y_true: np.ndarray) -> float:
        """Compute prediction bias B = E[f(x)|Y=1] - 0.5"""
        minority_mask = y_true == 1
        if minority_mask.sum() == 0:
            return 0.0
        return predictions[minority_mask].mean() - 0.5

    @staticmethod
    def compute_heterogeneity(biases: List[float]) -> float:
        """Compute bias heterogeneity H = Var(B_i)"""
        return np.var(biases)

    @staticmethod
    def compute_sigma(predictions: np.ndarray, y_true: np.ndarray) -> float:
        """Compute std of predictions on minority class."""
        minority_mask = y_true == 1
        if minority_mask.sum() < 2:
            return 0.1
        return predictions[minority_mask].std()

    @staticmethod
    def compute_threshold_H_star(
        sigma: float,
        B_bar: float,
        epsilon: float = 0.05,
        baseline_recall: float = None
    ) -> float:
        """
        Compute critical threshold H* (refined formula).

        H*_refined = H*_original × ceiling_penalty

        The ceiling penalty accounts for diminishing returns when
        baseline recall is already high.
        """
        if sigma < 1e-6:
            return float('inf')

        z = B_bar / sigma
        phi_z = norm.pdf(z)

        if phi_z < 1e-6:
            return float('inf')

        # Original formula
        H_star_original = (epsilon * sigma / phi_z) ** 2

        # Apply ceiling penalty if baseline_recall is provided
        if baseline_recall is not None:
            room = 1.0 - baseline_recall
            if room < 0.1:
                ceiling_penalty = 10.0
            elif room < 0.2:
                ceiling_penalty = 3.0
            elif room < 0.3:
                ceiling_penalty = 1.5
            else:
                ceiling_penalty = 1.0
            return H_star_original * ceiling_penalty

        return H_star_original

    @staticmethod
    def predict_recall_improvement(H: float, sigma: float, B_bar: float) -> float:
        """Predict recall improvement from Theorem 5."""
        if sigma < 1e-6 or H < 0:
            return 0.0
        z = B_bar / sigma
        phi_z = norm.pdf(z)
        return (np.sqrt(H) / sigma) * phi_z


class AdaptiveSpecialistEnsemble(BaseEstimator, ClassifierMixin):
    """
    Adaptive Specialist Ensemble (ASE)

    Optimizes specialist configuration to maximize bias heterogeneity H
    subject to maintaining minimum accuracy.
    """

    def __init__(
        self,
        base_estimator=None,
        K_max: int = 5,
        candidate_ratios: List[float] = None,
        n_per_type: int = 3,
        min_accuracy: str = 'auto',
        epsilon_H: float = 0.001,
        epsilon_recall: float = 0.05,
        val_size: float = 0.2,
        random_state: int = None
    ):
        self.base_estimator = base_estimator or RandomForestClassifier(
            n_estimators=100, random_state=random_state
        )
        self.K_max = K_max
        self.candidate_ratios = candidate_ratios or [1.0, 2.0, 3.0, 5.0, 9.0, float('inf')]
        self.n_per_type = n_per_type
        self.min_accuracy = min_accuracy
        self.epsilon_H = epsilon_H
        self.epsilon_recall = epsilon_recall
        self.val_size = val_size
        self.random_state = random_state

    def _create_specialist_data(self, X, y, ratio):
        """Create training data for a specialist with given ratio."""
        rng = np.random.RandomState(self.random_state)
        minority_idx = np.where(y == 1)[0]
        majority_idx = np.where(y == 0)[0]
        n_minority = len(minority_idx)

        if ratio == float('inf'):
            n_majority = len(majority_idx)
        else:
            n_majority = min(int(n_minority * ratio), len(majority_idx))

        sampled_majority_idx = rng.choice(majority_idx, size=n_majority, replace=False)
        selected_idx = np.concatenate([minority_idx, sampled_majority_idx])
        rng.shuffle(selected_idx)
        return X[selected_idx], y[selected_idx]

    def _train_specialist_type(self, X, y, ratio):
        """Train n_per_type classifiers for a specialist type."""
        classifiers = []
        for i in range(self.n_per_type):
            X_spec, y_spec = self._create_specialist_data(X, y, ratio)
            clf = clone(self.base_estimator)
            if hasattr(clf, 'random_state'):
                clf.random_state = (self.random_state or 0) + i
            clf.fit(X_spec, y_spec)
            classifiers.append(clf)
        return classifiers

    def _compute_config_metrics(self, config, X_train, y_train, X_val, y_val):
        """Train ensemble with config and compute metrics."""
        specialists = []
        for spec_config in config.specialists:
            classifiers = self._train_specialist_type(X_train, y_train, spec_config.ratio)
            specialists.append(classifiers)

        # Compute biases
        biases = []
        for type_classifiers in specialists:
            type_preds = np.mean([
                clf.predict_proba(X_val)[:, 1] for clf in type_classifiers
            ], axis=0)
            bias = TheoreticalAnalysis.compute_bias(type_preds, y_val)
            biases.append(bias)

        H = TheoreticalAnalysis.compute_heterogeneity(biases)

        # Ensemble predictions
        all_preds = []
        for type_classifiers in specialists:
            for clf in type_classifiers:
                all_preds.append(clf.predict_proba(X_val)[:, 1])

        ensemble_pred = np.mean(all_preds, axis=0)
        ensemble_labels = (ensemble_pred > 0.5).astype(int)

        return {
            'specialists': specialists,
            'biases': biases,
            'H': H,
            'accuracy': accuracy_score(y_val, ensemble_labels),
            'f1': f1_score(y_val, ensemble_labels),
            'recall': recall_score(y_val, ensemble_labels),
            'predictions': ensemble_pred
        }

    def _greedy_search(self, X_train, y_train, X_val, y_val, tau):
        """Greedy search for optimal configuration."""
        # Start with extreme specialists
        initial_ratios = [1.0, float('inf')]
        current_config = EnsembleConfig([
            SpecialistConfig(r, self.n_per_type) for r in initial_ratios
        ])

        best_metrics = self._compute_config_metrics(
            current_config, X_train, y_train, X_val, y_val
        )
        best_config = current_config

        available_ratios = [r for r in self.candidate_ratios if r not in initial_ratios]

        for k in range(3, self.K_max + 1):
            best_gain = 0
            best_new_ratio = None
            best_new_metrics = None
            best_new_config = None

            for ratio in available_ratios:
                trial_ratios = sorted(best_config.ratios + [ratio])
                trial_config = EnsembleConfig([
                    SpecialistConfig(r, self.n_per_type) for r in trial_ratios
                ])

                trial_metrics = self._compute_config_metrics(
                    trial_config, X_train, y_train, X_val, y_val
                )

                H_gain = trial_metrics['H'] - best_metrics['H']

                if H_gain > best_gain and trial_metrics['accuracy'] >= tau:
                    best_gain = H_gain
                    best_new_ratio = ratio
                    best_new_metrics = trial_metrics
                    best_new_config = trial_config

            if best_gain < self.epsilon_H:
                break

            best_config = best_new_config
            best_metrics = best_new_metrics
            available_ratios.remove(best_new_ratio)

            if not available_ratios:
                break

        return best_config, best_metrics

    def _optimize_weights(self, specialists, X_val, y_val):
        """
        Optimize weights for specialist types.

        Uses theory-guided bias-proportional weights that favor
        specialists with higher minority-class bias.
        """
        K = len(specialists)
        biases = []
        type_predictions = []

        for type_classifiers in specialists:
            type_pred = np.mean([
                clf.predict_proba(X_val)[:, 1] for clf in type_classifiers
            ], axis=0)
            type_predictions.append(type_pred)
            bias = TheoreticalAnalysis.compute_bias(type_pred, y_val)
            biases.append(bias)

        # THEORY-GUIDED: Bias-proportional weights
        # Higher bias = better at detecting minority → higher weight
        biases_arr = np.array(biases)

        # Shift biases to be positive and weight proportionally
        # Specialist with highest bias gets highest weight
        shifted_biases = biases_arr - biases_arr.min() + 0.1
        weights_theory = shifted_biases / shifted_biases.sum()

        # Also try F1-optimized weights for comparison
        best_f1 = 0
        best_weights_f1 = weights_theory.copy()

        if K == 2:
            for w1 in np.linspace(0.1, 0.9, 17):
                w = np.array([w1, 1 - w1])
                pred = sum(wi * p for wi, p in zip(w, type_predictions))
                f1 = f1_score(y_val, (pred > 0.5).astype(int))
                if f1 > best_f1:
                    best_f1 = f1
                    best_weights_f1 = w
        elif K == 3:
            for w1 in np.linspace(0.1, 0.8, 8):
                for w2 in np.linspace(0.1, 0.9 - w1, 8):
                    w3 = 1 - w1 - w2
                    if w3 < 0.05:
                        continue
                    w = np.array([w1, w2, w3])
                    pred = sum(wi * p for wi, p in zip(w, type_predictions))
                    f1 = f1_score(y_val, (pred > 0.5).astype(int))
                    if f1 > best_f1:
                        best_f1 = f1
                        best_weights_f1 = w

        # DECISION: Use theory-guided weights (bias-proportional)
        # This ensures we favor minority-specialist for better recall
        # F1-optimized weights often sacrifice recall for precision
        return weights_theory

    def fit(self, X, y):
        """Fit the Adaptive Specialist Ensemble."""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_size, stratify=y, random_state=self.random_state
        )

        # Determine minimum accuracy threshold
        if self.min_accuracy == 'auto':
            X_bal, y_bal = self._create_specialist_data(X_train, y_train, 1.0)
            baseline_clf = clone(self.base_estimator)
            baseline_clf.fit(X_bal, y_bal)
            baseline_pred = baseline_clf.predict(X_val)
            tau = accuracy_score(y_val, baseline_pred) * 0.95
        else:
            tau = self.min_accuracy

        # Greedy search
        self.config_, metrics = self._greedy_search(X_train, y_train, X_val, y_val, tau)
        self.specialists_ = metrics['specialists']

        # Theoretical analysis
        sigma = TheoreticalAnalysis.compute_sigma(metrics['predictions'], y_val)
        B_bar = np.mean(metrics['biases'])
        H_star = TheoreticalAnalysis.compute_threshold_H_star(sigma, B_bar, self.epsilon_recall)

        self.H_ = metrics['H']
        self.H_star_ = H_star
        self.analysis_ = {
            'biases': metrics['biases'],
            'sigma': sigma,
            'B_bar': B_bar,
            'H': metrics['H'],
            'H_star': H_star,
            'H_exceeds_threshold': metrics['H'] > H_star,
            'predicted_recall_improvement': TheoreticalAnalysis.predict_recall_improvement(
                metrics['H'], sigma, B_bar
            ),
            'validation_f1': metrics['f1'],
            'validation_recall': metrics['recall']
        }

        # Optimize weights
        self.weights_ = self._optimize_weights(self.specialists_, X_val, y_val)

        # Retrain on full data
        self.specialists_ = []
        for spec_config in self.config_.specialists:
            classifiers = self._train_specialist_type(X, y, spec_config.ratio)
            self.specialists_.append(classifiers)

        return self

    def predict_proba(self, X):
        """Predict class probabilities."""
        type_predictions = []
        for type_classifiers in self.specialists_:
            type_pred = np.mean([
                clf.predict_proba(X)[:, 1] for clf in type_classifiers
            ], axis=0)
            type_predictions.append(type_pred)

        ensemble_pred = sum(w * p for w, p in zip(self.weights_, type_predictions))
        return np.column_stack([1 - ensemble_pred, ensemble_pred])

    def predict(self, X):
        """Predict class labels."""
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def get_config_summary(self) -> str:
        """Return human-readable configuration summary."""
        return "\n".join([
            "Adaptive Specialist Ensemble Configuration:",
            f"  Specialist types: {self.config_.K}",
            f"  Ratios: {self.config_.ratios}",
            f"  Weights: {self.weights_.round(3)}",
            f"  H: {self.H_:.4f}",
            f"  H*: {self.H_star_:.4f}",
            f"  H > H*: {self.H_ > self.H_star_}"
        ])


class FixedSpecialistEnsemble(BaseEstimator, ClassifierMixin):
    """Fixed Specialist Ensemble with ratios 1:1, 1:3, 1:inf"""

    def __init__(self, base_estimator=None, n_per_type=3, random_state=None):
        self.base_estimator = base_estimator or RandomForestClassifier(
            n_estimators=100, random_state=random_state
        )
        self.n_per_type = n_per_type
        self.random_state = random_state
        self.ratios = [1.0, 3.0, float('inf')]
        self.type_weights = np.array([1.5, 1.2, 0.8])
        self.type_weights = self.type_weights / self.type_weights.sum()

    def fit(self, X, y):
        self.specialists_ = []
        rng = np.random.RandomState(self.random_state)
        minority_idx = np.where(y == 1)[0]
        majority_idx = np.where(y == 0)[0]
        n_minority = len(minority_idx)

        for ratio in self.ratios:
            type_classifiers = []
            for i in range(self.n_per_type):
                if ratio == float('inf'):
                    n_majority = len(majority_idx)
                else:
                    n_majority = min(int(n_minority * ratio), len(majority_idx))

                sampled_maj = rng.choice(majority_idx, size=n_majority, replace=False)
                selected_idx = np.concatenate([minority_idx, sampled_maj])
                rng.shuffle(selected_idx)

                clf = clone(self.base_estimator)
                clf.fit(X[selected_idx], y[selected_idx])
                type_classifiers.append(clf)
            self.specialists_.append(type_classifiers)
        return self

    def predict_proba(self, X):
        type_predictions = [
            np.mean([clf.predict_proba(X)[:, 1] for clf in tc], axis=0)
            for tc in self.specialists_
        ]
        ensemble_pred = sum(w * p for w, p in zip(self.type_weights, type_predictions))
        return np.column_stack([1 - ensemble_pred, ensemble_pred])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def compare_fixed_vs_adaptive(X, y, n_splits=5, random_state=42):
    """
    Compare fixed and adaptive specialist ensembles using cross-validation.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,)
    n_splits : int, number of CV folds
    random_state : int

    Returns
    -------
    results : list of dicts with metrics for each fold/method
    """
    from sklearn.model_selection import StratifiedKFold

    results = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fixed ensemble
        fixed = FixedSpecialistEnsemble(random_state=random_state)
        fixed.fit(X_train, y_train)
        fixed_pred = fixed.predict(X_test)

        # Adaptive ensemble
        adaptive = AdaptiveSpecialistEnsemble(random_state=random_state)
        adaptive.fit(X_train, y_train)
        adaptive_pred = adaptive.predict(X_test)

        results.append({
            'fold': fold,
            'method': 'Fixed',
            'f1': f1_score(y_test, fixed_pred),
            'recall': recall_score(y_test, fixed_pred),
            'precision': precision_score(y_test, fixed_pred),
            'K': 3,
            'H': None,
            'H_star': None
        })

        results.append({
            'fold': fold,
            'method': 'Adaptive',
            'f1': f1_score(y_test, adaptive_pred),
            'recall': recall_score(y_test, adaptive_pred),
            'precision': precision_score(y_test, adaptive_pred),
            'K': adaptive.config_.K,
            'H': adaptive.H_,
            'H_star': adaptive.H_star_
        })

    return results


def summarize_comparison(results):
    """Print summary of comparison results."""
    for method in ['Fixed', 'Adaptive']:
        method_results = [r for r in results if r['method'] == method]
        mean_f1 = np.mean([r['f1'] for r in method_results])
        std_f1 = np.std([r['f1'] for r in method_results])
        mean_recall = np.mean([r['recall'] for r in method_results])
        print(f"{method:10s}: F1={mean_f1:.3f}±{std_f1:.3f}, Recall={mean_recall:.3f}")

        if method == 'Adaptive':
            H_values = [r['H'] for r in method_results if r['H'] is not None]
            H_star_values = [r['H_star'] for r in method_results if r['H_star'] is not None]
            exceeds = sum(1 for h, hs in zip(H_values, H_star_values) if h > hs)
            print(f"           H > H* in {exceeds}/{len(H_values)} folds")


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_redundant=5,
        weights=[0.9, 0.1], random_state=42
    )

    print("Dataset: 1000 samples, IR = 9:1")
    print("=" * 50)

    # Train and show ASE config
    ase = AdaptiveSpecialistEnsemble(random_state=42)
    ase.fit(X, y)
    print(ase.get_config_summary())

    # Compare methods
    print("\n" + "=" * 50)
    print("5-Fold Cross-Validation Comparison:")
    print("-" * 50)
    results = compare_fixed_vs_adaptive(X, y)
    summarize_comparison(results)
