"""
Download datasets from multiple sources (no OpenML)
====================================================
Sources:
1. sklearn built-in datasets
2. UCI ML Repository (direct URLs)
3. GitHub repositories with CSV datasets
"""

import numpy as np
import pandas as pd
import urllib.request
import io
import os
from sklearn.datasets import load_breast_cancer, load_wine, load_digits, load_iris
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

DATASETS_DIR = os.path.dirname(os.path.abspath(__file__))


def download_file(url, timeout=30):
    """Download file from URL."""
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(req, timeout=timeout)
    return response.read().decode('utf-8')


def load_all_datasets():
    """Load datasets from multiple sources."""
    datasets = []

    print("=" * 70)
    print("DOWNLOADING DATASETS FROM MULTIPLE SOURCES")
    print("=" * 70)

    # =========================================================================
    # 1. SKLEARN DATASETS
    # =========================================================================
    print("\n[1/3] Loading sklearn datasets...")

    # Breast cancer
    print("  breast_cancer...", end=" ", flush=True)
    data = load_breast_cancer()
    X, y = data.data, 1 - data.target  # Malignant as minority (1)
    ir = np.bincount(y)[0] / np.bincount(y)[1]
    datasets.append({'name': 'breast_cancer', 'X': X, 'y': y, 'IR': ir,
                     'n_samples': len(y), 'n_minority': np.bincount(y)[1]})
    print(f"OK (n={len(y)}, IR={ir:.1f})")

    # Wine (class 2 as minority)
    print("  wine...", end=" ", flush=True)
    data = load_wine()
    X, y = data.data, (data.target == 2).astype(int)
    ir = np.bincount(y)[0] / np.bincount(y)[1]
    datasets.append({'name': 'wine', 'X': X, 'y': y, 'IR': ir,
                     'n_samples': len(y), 'n_minority': np.bincount(y)[1]})
    print(f"OK (n={len(y)}, IR={ir:.1f})")

    # Digits (9 vs rest)
    print("  digits_9...", end=" ", flush=True)
    data = load_digits()
    X, y = data.data, (data.target == 9).astype(int)
    ir = np.bincount(y)[0] / np.bincount(y)[1]
    datasets.append({'name': 'digits_9', 'X': X, 'y': y, 'IR': ir,
                     'n_samples': len(y), 'n_minority': np.bincount(y)[1]})
    print(f"OK (n={len(y)}, IR={ir:.1f})")

    # Digits (8 vs rest) - higher IR
    print("  digits_8...", end=" ", flush=True)
    X, y = data.data, (data.target == 8).astype(int)
    ir = np.bincount(y)[0] / np.bincount(y)[1]
    datasets.append({'name': 'digits_8', 'X': X, 'y': y, 'IR': ir,
                     'n_samples': len(y), 'n_minority': np.bincount(y)[1]})
    print(f"OK (n={len(y)}, IR={ir:.1f})")

    # Iris (setosa vs rest - easy)
    print("  iris_setosa...", end=" ", flush=True)
    data = load_iris()
    X, y = data.data, (data.target == 0).astype(int)
    ir = np.bincount(y)[0] / np.bincount(y)[1]
    datasets.append({'name': 'iris_setosa', 'X': X, 'y': y, 'IR': ir,
                     'n_samples': len(y), 'n_minority': np.bincount(y)[1]})
    print(f"OK (n={len(y)}, IR={ir:.1f})")

    # Digits variants with higher IR
    print("  digits_1...", end=" ", flush=True)
    data = load_digits()
    X, y = data.data, (data.target == 1).astype(int)
    ir = np.bincount(y)[0] / np.bincount(y)[1]
    datasets.append({'name': 'digits_1', 'X': X, 'y': y, 'IR': ir,
                     'n_samples': len(y), 'n_minority': np.bincount(y)[1]})
    print(f"OK (n={len(y)}, IR={ir:.1f})")

    # Digits (0 vs rest)
    print("  digits_0...", end=" ", flush=True)
    X, y = data.data, (data.target == 0).astype(int)
    ir = np.bincount(y)[0] / np.bincount(y)[1]
    datasets.append({'name': 'digits_0', 'X': X, 'y': y, 'IR': ir,
                     'n_samples': len(y), 'n_minority': np.bincount(y)[1]})
    print(f"OK (n={len(y)}, IR={ir:.1f})")

    # =========================================================================
    # 2. GITHUB DATASETS (Jason Brownlee's ML datasets)
    # =========================================================================
    print("\n[2/3] Downloading from GitHub...")

    github_datasets = [
        # (url, name, has_header, label_col, separator)
        ('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv',
         'pima_diabetes', False, -1, ','),
        # Kaggle datasets (public, no auth required)
        ('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv',
         'titanic', True, 1, ','),
        # Heart disease
        ('https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv',
         'heart_disease', True, -1, ','),
        # Sonar (Mines vs Rocks)
        ('https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv',
         'sonar', False, -1, ','),
        # Ionosphere
        ('https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv',
         'ionosphere', False, -1, ','),
        # German Credit
        ('https://raw.githubusercontent.com/jbrownlee/Datasets/master/german.csv',
         'german_credit', False, -1, ','),
        # Hepatitis
        ('https://raw.githubusercontent.com/jbrownlee/Datasets/master/hepatitis.csv',
         'hepatitis', False, -1, ','),
        # Horse Colic
        ('https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv',
         'horse_colic', False, -1, ','),
    ]

    for url, name, has_header, label_col, sep in github_datasets:
        try:
            print(f"  {name}...", end=" ", flush=True)
            content = download_file(url, timeout=15)

            # Parse CSV
            lines = [l for l in content.strip().split('\n') if l.strip()]
            if has_header:
                lines = lines[1:]

            data = []
            for line in lines:
                parts = line.split(sep)
                # Convert to float, handle missing values
                row = []
                for p in parts:
                    p = p.strip()
                    if p == '?' or p == '' or p == 'NA':
                        row.append(np.nan)
                    else:
                        try:
                            row.append(float(p))
                        except:
                            # Encode categorical as numbers
                            row.append(hash(p) % 1000)
                data.append(row)

            data = np.array(data, dtype=float)

            # Remove rows with too many NaNs
            nan_mask = np.isnan(data).sum(axis=1) < data.shape[1] * 0.3
            data = data[nan_mask]

            # Impute remaining NaNs
            for i in range(data.shape[1]):
                col = data[:, i]
                nan_idx = np.isnan(col)
                if nan_idx.any():
                    col[nan_idx] = np.nanmedian(col)

            X = data[:, :label_col] if label_col != -1 else data[:, :-1]
            y = data[:, label_col].astype(int)

            # Binarize if needed
            unique_y = np.unique(y[~np.isnan(y)])
            if len(unique_y) > 2:
                counts = np.bincount(y.astype(int))
                minority = np.argmin(counts)
                y = (y == minority).astype(int)
            else:
                y = (y == unique_y.max()).astype(int)

            # Ensure minority is 1
            counts = np.bincount(y)
            if counts[0] < counts[1]:
                y = 1 - y
                counts = np.bincount(y)

            ir = counts[0] / max(counts[1], 1)

            if counts[1] >= 10:
                datasets.append({'name': name, 'X': X, 'y': y, 'IR': ir,
                                'n_samples': len(y), 'n_minority': counts[1]})
                print(f"OK (n={len(y)}, IR={ir:.1f})")
            else:
                print(f"SKIPPED (only {counts[1]} minority)")

        except Exception as e:
            print(f"FAILED - {str(e)[:40]}")

    # =========================================================================
    # 3. UCI DATASETS (direct download)
    # =========================================================================
    print("\n[3/3] Downloading from UCI...")

    # (url, name, label_col, delimiter)
    uci_datasets = [
        ('https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data',
         'spambase', -1, ','),
        ('https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt',
         'banknote', -1, ','),
        ('https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data',
         'transfusion', -1, ','),
        ('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat',
         'australian_credit', -1, ' '),
    ]

    for url, name, label_col, delim in uci_datasets:
        try:
            print(f"  {name}...", end=" ", flush=True)
            content = download_file(url, timeout=15)

            lines = content.strip().split('\n')
            # Skip header if exists
            first_line = lines[0].replace(',', '').replace('.', '').replace('-', '').replace(' ', '').strip()
            if not first_line.replace('g', '').replace('h', '').isdigit():  # g,h for magic dataset
                lines = lines[1:]

            data = np.genfromtxt(io.StringIO('\n'.join(lines)),
                                delimiter=delim if delim != ' ' else None,
                                dtype=float, filling_values=np.nan)
            data = data[~np.isnan(data).any(axis=1)]

            X = data[:, :label_col]
            y = data[:, label_col].astype(int)

            # Ensure minority is 1
            counts = np.bincount(y)
            if counts[0] < counts[1]:
                y = 1 - y
                counts = np.bincount(y)

            ir = counts[0] / max(counts[1], 1)

            if counts[1] >= 10:
                datasets.append({'name': name, 'X': X, 'y': y, 'IR': ir,
                                'n_samples': len(y), 'n_minority': counts[1]})
                print(f"OK (n={len(y)}, IR={ir:.1f})")
            else:
                print(f"SKIPPED (only {counts[1]} minority)")

        except Exception as e:
            print(f"FAILED - {str(e)[:40]}")

    # =========================================================================
    # 4. KEEL HIGH-IR DATASETS
    # =========================================================================
    print("\n[4/4] Downloading KEEL high-IR datasets...")

    import zipfile

    keel_datasets = [
        ('https://sci2s.ugr.es/keel/dataset/data/imbalanced/glass5.zip', 'glass5'),
        ('https://sci2s.ugr.es/keel/dataset/data/imbalanced/yeast4.zip', 'yeast4'),
        ('https://sci2s.ugr.es/keel/dataset/data/imbalanced/yeast5.zip', 'yeast5'),
        ('https://sci2s.ugr.es/keel/dataset/data/imbalanced/yeast6.zip', 'yeast6'),
        ('https://sci2s.ugr.es/keel/dataset/data/imbalanced/ecoli4.zip', 'ecoli4'),
    ]

    for url, name in keel_datasets:
        try:
            print(f"  {name}...", end=" ", flush=True)
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            response = urllib.request.urlopen(req, timeout=15)
            zip_data = response.read()

            # Extract .dat file from zip
            with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                dat_files = [f for f in zf.namelist() if f.endswith('.dat') and 'tra' not in f.lower() and 'tst' not in f.lower()]
                if not dat_files:
                    dat_files = [f for f in zf.namelist() if f.endswith('.dat')]

                if dat_files:
                    with zf.open(dat_files[0]) as f:
                        content = f.read().decode('utf-8')

                    # Parse KEEL format
                    lines = content.strip().split('\n')
                    data_started = False
                    data_lines = []

                    for line in lines:
                        if '@data' in line.lower():
                            data_started = True
                            continue
                        if data_started and line.strip() and not line.startswith('@'):
                            data_lines.append(line.strip())

                    # Parse data
                    rows = []
                    for line in data_lines:
                        parts = line.split(',')
                        try:
                            features = [float(p.strip()) for p in parts[:-1]]
                            label = parts[-1].strip()
                            rows.append((features, label))
                        except:
                            continue

                    if rows:
                        X = np.array([r[0] for r in rows])
                        labels = [r[1] for r in rows]
                        le = LabelEncoder()
                        y = le.fit_transform(labels)

                        # Ensure minority is 1
                        counts = np.bincount(y)
                        if counts[0] < counts[1]:
                            y = 1 - y
                            counts = np.bincount(y)

                        ir = counts[0] / max(counts[1], 1)

                        if counts[1] >= 10:
                            datasets.append({'name': name, 'X': X, 'y': y, 'IR': ir,
                                           'n_samples': len(y), 'n_minority': counts[1]})
                            print(f"OK (n={len(y)}, IR={ir:.1f})")
                        else:
                            print(f"SKIPPED (only {counts[1]} minority)")
                    else:
                        print("FAILED - No data rows")
                else:
                    print("FAILED - No .dat file found")

        except Exception as e:
            print(f"FAILED - {str(e)[:40]}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"TOTAL: {len(datasets)} datasets loaded successfully")
    print("=" * 70)

    for i, d in enumerate(datasets, 1):
        print(f"  {i:2}. {d['name']:20} n={d['n_samples']:5}, IR={d['IR']:6.1f}, minority={d['n_minority']}")

    return datasets


if __name__ == "__main__":
    datasets = load_all_datasets()

    # Save summary
    summary = pd.DataFrame([{
        'name': d['name'],
        'n_samples': d['n_samples'],
        'n_features': d['X'].shape[1],
        'n_minority': d['n_minority'],
        'IR': d['IR']
    } for d in datasets])

    summary.to_csv('datasets_summary.csv', index=False)
    print(f"\nSummary saved to datasets_summary.csv")
