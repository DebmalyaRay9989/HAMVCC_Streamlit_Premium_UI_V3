"""Manuscript-aligned, exploratory diagnostics; never modifies reference evidence."""
import numpy as np
import pandas as pd
from core import gate, runs

def cross_environment(metric='ARI', tolerance=1e-10):
    if metric not in ('ARI','NMI') or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Invalid metric or tolerance')
    keys=['condition','split','seed','method']
    a=runs('reference')[keys+[metric]].rename(columns={metric:'linux'})
    b=runs('windows')[keys+[metric]].rename(columns={metric:'windows'})
    joined=a.merge(b,on=keys,validate='one_to_one',indicator=True)
    if not (joined._merge=='both').all() or len(joined)!=len(a) or len(joined)!=len(b):
        raise ValueError('Unmatched experimental keys; comparison refused')
    joined['difference']=joined.windows-joined.linux
    joined['absolute_difference']=joined.difference.abs()
    joined['discrepant']=joined.absolute_difference>tolerance
    return joined.drop(columns='_merge')

def gate_jacobian(reliability, mask, kappa):
    weights=gate(reliability,mask,kappa)
    jac=2*kappa*(np.diag(weights)-np.outer(weights,weights))
    return weights,jac

def coordinate_experiment(seed=42,n=90,d=5,kappa=.752):
    """Synthetic orthogonal-coordinate demonstration, not a manuscript rerun."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from scipy.spatial.distance import pdist
    if not 20<=n<=500 or not 2<=d<=32 or not 0<=kappa<=10:
        raise ValueError('Out-of-range demonstration parameters')
    rng=np.random.default_rng(seed)
    latent=rng.normal(size=(n,d))
    views=[latent+rng.normal(scale=.45,size=(n,d)) for _ in range(3)]
    reliability=np.array([.85,.55,.3]);w=gate(reliability,[1,1,1],kappa)
    rotated=[]
    for v in views:
        q,_=np.linalg.qr(rng.normal(size=(d,d)))
        rotated.append(v@q)
    outputs=[]
    for name,fn in [('Unaligned sum',lambda v:sum(w[i]*v[i] for i in range(3))),
                    ('Masked concatenation',lambda v:np.concatenate(v,axis=1)),
                    ('Gated concatenation',lambda v:np.concatenate([3*w[i]*v[i] for i in range(3)],axis=1))]:
        original=fn(views); changed=fn(rotated)
        labels0=KMeans(n_clusters=3,n_init=10,random_state=seed).fit_predict(original)
        labels1=KMeans(n_clusters=3,n_init=10,random_state=seed).fit_predict(changed)
        outputs.append({'Fusion':name,'Max pairwise distance change':float(np.max(np.abs(pdist(original)-pdist(changed)))),
                        'Partition ARI vs original':float(adjusted_rand_score(labels0,labels1))})
    return pd.DataFrame(outputs)
