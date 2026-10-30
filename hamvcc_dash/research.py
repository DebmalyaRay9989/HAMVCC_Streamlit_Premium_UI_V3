"""Exploratory HAMV-CC research experiments; never modifies bundled evidence."""
import hashlib, json, platform, sys, time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from core import gate, floor_strength


def prepare_views(frame, columns, mask_prefix='available_', max_rows=2000):
    """Columns grouped by view prefix, e.g. visual__0, text__0; NaN means missing."""
    if len(frame)>max_rows: raise ValueError('Maximum 2,000 documents')
    groups={}
    for col in columns:
        if '__' not in col: continue
        name=col.split('__',1)[0]
        if name.startswith('available_') or name in ('label','document_id'): continue
        groups.setdefault(name,[]).append(col)
    if not 2<=len(groups)<=6: raise ValueError('Require 2–6 views with columns view__feature')
    views=[]; masks=[]
    for name, cols in groups.items():
        a=frame[cols].apply(__import__('pandas').to_numeric,errors='coerce').to_numpy(dtype=float)
        finite=np.isfinite(a).all(axis=1)
        explicit=mask_prefix+name
        if explicit in frame:
            raw=frame[explicit].to_numpy()
            if not np.isin(raw,[0,1,False,True]).all(): raise ValueError('Availability must be binary')
            finite &= raw.astype(bool)
        a[~np.isfinite(a)]=0
        views.append(a);masks.append(finite)
    mask=np.array(masks).T
    if (~mask.any(axis=1)).any(): raise ValueError('All-missing rows are unassignable; remove or repair explicitly')
    return list(groups),views,mask


def corrupt(views,mask,missing=0.2,noise=0.2,seed=0,mode='gaussian'):
    rng=np.random.default_rng(seed)
    out=[x.copy() for x in views]
    newmask=mask.copy()
    for j,x in enumerate(out):
        drops=rng.random(len(x))<missing
        newmask[:,j]&=~drops
        if mode=='gaussian': x[:]+=rng.normal(0,noise,x.shape)
        elif mode=='dropout': x[:]*=(rng.random(x.shape)>=noise)
        elif mode=='scale': x[:]*=1+rng.normal(0,noise,(len(x),1))
        else: raise ValueError('Unknown corruption')
    # No silent repair: rows with all views missing are excluded and counted.
    valid=newmask.any(axis=1)
    return out,newmask,valid


def reliability_models(clean,corrupted,mask,train_idx,seed=0):
    """Per-view proxy regressor trained only on train partition; labels not used."""
    scores=np.zeros(mask.shape)
    models=[]
    for j,(x,y) in enumerate(zip(clean,corrupted)):
        den=np.maximum(np.linalg.norm(x,axis=1),1e-8)
        target=np.clip(1-np.linalg.norm(y-x,axis=1)/den,0,1)
        # Quality proxies from observable corrupted features only.
        features=np.column_stack([np.mean(abs(y),axis=1),np.std(y,axis=1),np.mean(y*y,axis=1),np.mean(y==0,axis=1),np.linalg.norm(y,axis=1)])
        ids=train_idx[mask[train_idx,j]]
        if len(ids)<10: raise ValueError('Too few available training observations for reliability fitting')
        model=HistGradientBoostingRegressor(max_iter=80,random_state=seed,min_samples_leaf=5)
        model.fit(features[ids],target[ids]);scores[:,j]=np.clip(model.predict(features),0,1)
        models.append(model)
    return scores,models


def fuse(views,mask,reliability,kappa=0.752,method='gated_concat'):
    n,m=mask.shape
    weights=np.stack([gate(reliability[i],mask[i].astype(int),kappa if method!='masked_concat' else 0) for i in range(n)])
    blocks=[]
    for j,x in enumerate(views):
        norms=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-9)
        z=x/norms
        blocks.append(z*(m*weights[:,j,None]))
    if method=='sum':
        if len(set(x.shape[1] for x in blocks))!=1: raise ValueError('Sum fusion requires equal-dimensional, aligned view spaces')
        return np.sum(blocks,axis=0),weights
    return np.concatenate(blocks,axis=1),weights


def experiment(frame,missing=.2,noise=.2,k=4,seed=0,mode='gaussian',methods=None):
    methods=methods or ['masked_concat','gated_concat','sum']
    columns=list(frame.columns)
    names,views,mask=prepare_views(frame,columns)
    rng=np.random.default_rng(seed)
    indices=rng.permutation(len(frame));cut=max(10,int(.6*len(frame)))
    train=indices[:cut]
    perturbed,mask2,valid=corrupt(views,mask,missing,noise,seed,mode)
    if valid.sum()<k+2: raise ValueError('Insufficient assignable rows')
    scores,_=reliability_models(views,perturbed,mask2,train,seed)
    # Floor depends on maximum number of available views.
    kap=floor_strength(.1,len(views)) if len(views)<10 else 0
    results=[];assignments={};weights={}
    labels=frame['label'].to_numpy() if 'label' in frame else None
    for method in methods:
        try:
            z,w=fuse([x[valid] for x in perturbed],mask2[valid],scores[valid],kap,method)
            pred=KMeans(n_clusters=k,random_state=seed,n_init=5).fit_predict(z)
            row={'method':method,'n_assignable':int(valid.sum()),'n_unassignable':int((~valid).sum()),'silhouette':float(silhouette_score(z,pred)) if len(np.unique(pred))>1 and len(z)>k else None}
            if labels is not None:
                row['ARI']=float(adjusted_rand_score(labels[valid],pred))
                row['NMI']=float(normalized_mutual_info_score(labels[valid],pred))
            results.append(row);assignments[method]=pred.tolist();weights[method]=w.mean(axis=0).tolist()
        except ValueError as exc: results.append({'method':method,'error':str(exc)})
    manifest={'kind':'exploratory_new_execution','created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(), 'seed':seed,'missing':missing,'noise':noise,'corruption':mode,'k':k,'views':{n:list(x.shape) for n,x in zip(names,views)},'rows':len(frame),'input_sha256':hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),'python':sys.version.split()[0],'numpy':np.__version__,'sklearn':__import__('sklearn').__version__,'platform':platform.platform(),'gate_floor':.1,'kappa':kap,'label_usage':'final evaluation only; never fitting'}
    return {'results':results,'assignments':assignments,'mean_weights':weights,'manifest':manifest}


def ablate(frame,k=4,seed=0):
    out=[]
    for missing,noise in [(0,0),(.2,0),(.2,.1),(.2,.2),(.4,.2)]:
        for s in [seed,seed+1,seed+2]:
            run=experiment(frame,missing,noise,k,s,methods=['masked_concat','gated_concat','sum'])
            for r in run['results']: out.append(dict(missing=missing,noise=noise,seed=s,**r))
    return out
