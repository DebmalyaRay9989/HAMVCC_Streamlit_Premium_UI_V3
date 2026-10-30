"""Numerical gate, immutable result loaders, and bounded exploratory clustering."""
from functools import lru_cache
from pathlib import Path
import base64
import io
import json
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / 'data'
ENVIRONMENTS = {'reference': ('Linux reference', 'reference_results_v2.json', 'reference_analysis_v2.json'),
                'windows': ('Windows rerun', 'windows_results.json', 'windows_analysis.json')}
CONDITIONS = ['M0+D50', 'M20+D50', 'M40+D50', 'M20+clean']

@lru_cache(maxsize=16)
def load_json(name):
    return json.loads((DATA / name).read_text(encoding='utf-8-sig'))

@lru_cache(maxsize=16)
def load_csv(name):
    return pd.read_csv(DATA / name, encoding='utf-8-sig')

@lru_cache(maxsize=2)
def runs(env):
    rows = []
    for run in load_json(ENVIRONMENTS[env][1])['runs']:
        for method, scores in run['m'].items():
            rows.append(dict(environment=env, condition=run['cond'], split=run['split'],
                             seed=run['seed'], method=method, ARI=scores[0], NMI=scores[1]))
    return pd.DataFrame(rows)

def gate(reliability, availability, kappa):
    r = np.asarray(reliability, dtype=float)
    raw_mask = np.asarray(availability)
    if r.ndim != 1 or raw_mask.shape != r.shape or not np.isin(raw_mask, [0, 1]).all():
        raise ValueError('Reliability and binary availability must have matching one-dimensional shapes.')
    if not np.isfinite(r).all() or ((r < 0) | (r > 1)).any():
        raise ValueError('Reliability scores must be finite values between 0 and 1.')
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError('Gate strength must be finite and nonnegative.')
    mask = raw_mask.astype(bool)
    if not mask.any():
        raise ValueError('Unassignable: all views are missing. Select at least one available view.')
    logits = kappa * (2*r[mask] - 1)
    weights = np.zeros_like(r)
    weights[mask] = np.exp(logits-logits.max())
    return weights/weights.sum()

def floor_strength(rho, count):
    if count < 2:
        return 0.0
    if not 0 < rho <= 1/count:
        raise ValueError('The requested floor must be at most 1 / available views.')
    return float(0.5*np.log((1/rho-1)/(count-1)))

def parse_upload(contents, filename):
    if not filename or not filename.lower().endswith('.csv'):
        raise ValueError('Choose a UTF-8 CSV file.')
    if len(contents) > 4_100_000:
        raise ValueError('The CSV upload limit is 3 MB.')
    try:
        raw = base64.b64decode(contents.split(',', 1)[1], validate=True)
        if len(raw) > 3_000_000:
            raise ValueError('The CSV upload limit is 3 MB.')
        df = pd.read_csv(io.StringIO(raw.decode('utf-8-sig')), nrows=2001)
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, IndexError) as exc:
        raise ValueError('Could not read this file as a UTF-8 CSV.') from exc
    if not 4 <= len(df) <= 2000 or len(df.columns) > 64:
        raise ValueError('Use 4–2,000 rows and at most 64 columns.')
    if 'document_id' not in df:
        df.insert(0, 'document_id', [f'row-{i+1}' for i in range(len(df))])
    df['document_id'] = df['document_id'].astype(str)
    if df.document_id.duplicated().any():
        raise ValueError('document_id must be unique.')
    return df

def numeric_features(df):
    excluded = {'document_index', 'bytes', 'original_width', 'original_height', 'cluster', 'label', 'target', 'document_id'}
    return [c for c in df.select_dtypes(include='number').columns if c not in excluded]

def cluster_features(df, columns, k, seed):
    # Import costly estimators only on explicit Run; no background training.
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    from threadpoolctl import threadpool_limits
    import time
    start = time.perf_counter()
    if not columns or any(c not in numeric_features(df) for c in columns):
        raise ValueError('Select at least one numeric feature; identifiers and labels are excluded.')
    if not 4 <= len(df) <= 2000 or len(columns) > 64:
        raise ValueError('Use 4–2,000 rows and at most 64 features.')
    if k is None or int(k) != k or not 2 <= int(k) <= min(12, len(df)-1):
        raise ValueError('K must be an integer from 2 to min(12, rows − 1).')
    if seed is None or int(seed) != seed or not 0 <= int(seed) <= 2**32-1:
        raise ValueError('Seed must be an integer between 0 and 4294967295.')
    x = df[columns].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('Selected features contain missing or infinite values. Clean the CSV first.')
    active = np.ptp(x, axis=0) > 0
    if not active.any():
        raise ValueError('All selected features are constant.')
    if len(np.unique(x[:, active], axis=0)) < k:
        raise ValueError('K exceeds the number of distinct feature rows.')
    with threadpool_limits(limits=1):
        z = StandardScaler().fit_transform(x[:, active])
        labels = KMeans(n_clusters=int(k), n_init=10, random_state=int(seed), algorithm='lloyd').fit_predict(z)
        pca = PCA(n_components=min(2, z.shape[1]), svd_solver='full')
        coords = pca.fit_transform(z)
        if coords.shape[1] == 1:
            coords = np.column_stack([coords, np.zeros(len(z))])
        # Bound pairwise metric work; this sampling is explicitly reported.
        sil = silhouette_score(z, labels, sample_size=min(500, len(z)), random_state=int(seed)) if len(z) <= 500 else None
        if len(z) > 500:
            # Include each cluster to avoid degenerate random subsamples.
            rng = np.random.default_rng(int(seed))
            first = [np.flatnonzero(labels == c)[0] for c in np.unique(labels)]
            rest = np.setdiff1d(np.arange(len(z)), first)
            ix = np.r_[first, rng.choice(rest, 500-len(first), replace=False)]
            sil = silhouette_score(z[ix], labels[ix])
        db = davies_bouldin_score(z, labels)
    out = pd.DataFrame({'document_id': df.document_id.astype(str), 'cluster': labels.astype(str), 'PC1': coords[:, 0], 'PC2': coords[:, 1]})
    info = dict(silhouette=float(sil), davies_bouldin=float(db), seconds=time.perf_counter()-start,
                rows=len(df), features=int(active.sum()), dropped_constants=int((~active).sum()),
                pca_variance=float(pca.explained_variance_ratio_.sum()), k=int(k), seed=int(seed),
                silhouette_rows=min(500, len(z)), feature_names=list(np.array(columns)[active]),
                protocol='StandardScaler → KMeans(n_init=10); full-SVD PCA for display only; exploratory, no semantic labels')
    return out, info
