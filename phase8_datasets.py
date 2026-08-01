"""
Phase 8: Dataset caching
========================
Loads the full benchmark (sklearn + GitHub + UCI + KEEL high-IR) once and
caches it to datasets_phase8.pkl so experiments do not depend on the network.

The KEEL server (sci2s.ugr.es) currently serves an expired SSL certificate;
we patch ssl's default context for that download only (public benchmark data,
content validated by the KEEL parser afterwards).
"""

import os
import pickle
import ssl
import urllib.request

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets_phase8.pkl')


def load_datasets(force_reload=False):
    if os.path.exists(CACHE_FILE) and not force_reload:
        with open(CACHE_FILE, 'rb') as f:
            datasets = pickle.load(f)
        print(f"Loaded {len(datasets)} datasets from cache ({CACHE_FILE})")
        return datasets

    # KEEL cert workaround: urlopen without context arg uses the default context
    _orig_urlopen = urllib.request.urlopen

    def _patched_urlopen(url, *args, **kwargs):
        if 'context' not in kwargs or kwargs['context'] is None:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            kwargs['context'] = ctx
        return _orig_urlopen(url, *args, **kwargs)

    urllib.request.urlopen = _patched_urlopen
    try:
        from download_datasets import load_all_datasets
        datasets = load_all_datasets()
    finally:
        urllib.request.urlopen = _orig_urlopen

    for d in datasets:
        d['n_features'] = d['X'].shape[1]

    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(datasets, f)
    print(f"\nCached {len(datasets)} datasets to {CACHE_FILE}")
    return datasets


if __name__ == "__main__":
    datasets = load_datasets(force_reload=True)
    print(f"\nTOTAL: {len(datasets)}")
    for d in sorted(datasets, key=lambda d: -d['IR']):
        print(f"  {d['name']:25s} n={d['n_samples']:6d} feat={d['n_features']:3d} IR={d['IR']:.1f}")
