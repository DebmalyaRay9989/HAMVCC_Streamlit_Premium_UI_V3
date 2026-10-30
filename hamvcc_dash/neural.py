"""HAMV-CC neural research implementation. New experiments, not manuscript reproductions.

Training uses only development rows. Test labels are never used for fitting,
checkpoint selection, or cluster-number selection. Missing views have zero paths.
"""
import copy, hashlib, json, os, platform, random, sys, tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np

@dataclass
class NeuralConfig:
    clusters: int = 4
    latent: int = 64
    hidden: int = 128
    epochs: int = 40
    batch_size: int = 128
    lr: float = 0.001
    gamma: float = 0.5
    floor: float = 0.1
    contrastive_temperature: float = 0.2
    lambda_instance: float = 1.0
    lambda_consistency: float = 1.0
    lambda_balance: float = 1.0
    augment_std: float = 0.08
    patience: int = 12
    seed: int = 17
    test_fraction: float = 0.2
    validation_fraction: float = 0.2
    checkpoint_dir: str = 'checkpoints'


def gate_strength(floor, views):
    if views < 2: return 0.0
    if not 0 < floor < 1/views: raise ValueError('floor must be between zero and 1/number_of_views')
    return 0.5 * np.log((1/floor-1)/(views-1))


def seed_everything(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def model_factory(dims, cfg):
    import torch
    from torch import nn
    from torch.nn import functional as F
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoders=nn.ModuleList([nn.Sequential(nn.Linear(d,cfg.hidden),nn.ReLU(),nn.Linear(cfg.hidden,cfg.latent)) for d in dims])
            self.residuals=nn.ModuleList([nn.Sequential(nn.Linear(cfg.latent,cfg.latent),nn.Tanh()) for _ in dims])
            self.projections=nn.ModuleList([nn.Linear(cfg.latent,cfg.latent,bias=False) for _ in dims])
            self.head=nn.Linear(cfg.latent,cfg.clusters)
            self.kappa=gate_strength(cfg.floor,len(dims))
        def forward(self,views,mask,reliability):
            if not torch.all(mask.sum(1)>0): raise ValueError('All-missing samples are unassignable')
            logits=self.kappa*(2*reliability.clamp(0,1)-1)
            logits=logits.masked_fill(~mask.bool(),torch.finfo(logits.dtype).min)
            alpha=torch.softmax(logits,dim=1)*mask
            alpha=alpha/alpha.sum(1,keepdim=True).clamp_min(1e-12)
            mean_rel=(reliability*mask).sum(1,keepdim=True)/mask.sum(1,keepdim=True)
            projected=[]; qviews=[]
            for j,x in enumerate(views):
                active=mask[:,j:j+1]
                h=self.encoders[j](x*active)*active
                residual=self.residuals[j](h)*active
                h=(h+cfg.gamma*(reliability[:,j:j+1]-mean_rel)*residual)*active
                p=F.normalize(self.projections[j](h)*active,dim=1)*active
                projected.append(p)
                qviews.append(torch.softmax(self.head(p),dim=1))
            # Neural model's learned projections establish common coordinates.
            fused=F.normalize(sum(alpha[:,j:j+1]*p for j,p in enumerate(projected)),dim=1)
            qf=torch.softmax(self.head(fused),dim=1)
            return {'fused':fused,'views':projected,'qf':qf,'qviews':qviews,'alpha':alpha}
    return Model()


def instance_loss(z1,z2,temp):
    """Symmetric NT-Xent with each sample's other augmentation as its positive."""
    import torch
    from torch.nn import functional as F
    n=z1.shape[0]
    if n<2: return (z1.sum()+z2.sum())*0
    z=F.normalize(torch.cat([z1,z2],0),dim=1)
    logits=z@z.T/temp
    logits.fill_diagonal_(torch.finfo(logits.dtype).min)
    positives=(torch.arange(2*n,device=z.device)+n)%(2*n)
    return F.cross_entropy(logits,positives)


def js_consistency(out,mask,reliability):
    import torch
    from torch.nn import functional as F
    qf=out['qf'].clamp_min(1e-8)
    total=qf.new_zeros(()); denom=(mask*reliability).sum().clamp_min(1e-8)
    for j,q in enumerate(out['qviews']):
        q=q.clamp_min(1e-8)
        mid=(q+qf)/2
        js=.5*((q*(q.log()-mid.log())).sum(1)+(qf*(qf.log()-mid.log())).sum(1))
        total=total+(js*mask[:,j]*reliability[:,j]).sum()
    return total/denom


def balance_loss(q):
    import torch
    p=q.clamp_min(1e-8); avg=p.mean(0).clamp_min(1e-8)
    return -(p*p.log()).sum(1).mean()+(avg*avg.log()).sum()


def train_neural(views,mask,reliability,k=4,epochs=40,seed=17,labels=None,
                 checkpoint_dir='checkpoints',config=None,run_test=False):
    """Fit train/validation only; test evaluation is explicit and single-shot.

    Returns JSON-safe metadata. Checkpoint contains fitted state, preprocessing,
    split IDs, configuration and hashes; no raw user dataset is embedded.
    """
    import torch
    from sklearn.model_selection import train_test_split
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
    cfg=config or NeuralConfig(clusters=int(k),epochs=int(epochs),seed=int(seed),checkpoint_dir=checkpoint_dir)
    seed_everything(cfg.seed)
    a=np.asarray(mask,dtype=bool);r=np.asarray(reliability,dtype=np.float32)
    x=[np.asarray(v,dtype=np.float32) for v in views]
    n=len(a);m=len(x)
    if not 2<=m<=6 or a.shape!=(n,m) or r.shape!=(n,m) or any(v.ndim!=2 or len(v)!=n for v in x): raise ValueError('Invalid multi-view shapes')
    if not np.isfinite(r).all() or any(not np.isfinite(v).all() for v in x): raise ValueError('Nonfinite input')
    if (a.sum(1)==0).any(): raise ValueError('Remove all-missing rows explicitly')
    if n<max(30,cfg.clusters*5): raise ValueError('Insufficient data for independent partitions')
    ids=np.arange(n)
    dev,test=train_test_split(ids,test_size=cfg.test_fraction,random_state=cfg.seed)
    train,val=train_test_split(dev,test_size=cfg.validation_fraction,random_state=cfg.seed+1)
    # Fit normalization only on available training observations; no test leakage.
    stats=[];normalized=[]
    for j,v in enumerate(x):
        available=v[train[a[train,j]]]
        if len(available)<2: raise ValueError('Not enough available training rows')
        mu=available.mean(0);sd=available.std(0);sd=np.maximum(sd,1e-6)
        z=np.clip((v-mu)/sd,-10,10);z[~a[:,j]]=0
        normalized.append(torch.tensor(z,dtype=torch.float32))
        stats.append({'mean':mu.tolist(),'std':sd.tolist()})
    mask_t=torch.tensor(a,dtype=torch.float32)
    rel_t=torch.tensor(np.clip(r,0,1),dtype=torch.float32)*mask_t
    model=model_factory([v.shape[1] for v in x],cfg)
    opt=torch.optim.Adam(model.parameters(),lr=cfg.lr)
    best=float('inf');best_state=None;best_epoch=0;wait=0;history=[]
    rng=np.random.default_rng(cfg.seed)
    for epoch in range(cfg.epochs):
        model.train(); losses=[]
        for batch in np.array_split(rng.permutation(train),max(1,int(np.ceil(len(train)/cfg.batch_size)))):
            if len(batch)<2: continue
            vb=[v[batch] for v in normalized];mb=mask_t[batch];rb=rel_t[batch]
            v1=[(v+torch.randn_like(v)*cfg.augment_std)*mb[:,j:j+1] for j,v in enumerate(vb)]
            v2=[(v+torch.randn_like(v)*cfg.augment_std)*mb[:,j:j+1] for j,v in enumerate(vb)]
            o1=model(v1,mb,rb);o2=model(v2,mb,rb)
            inst=instance_loss(o1['fused'],o2['fused'],cfg.contrastive_temperature)
            cons=.5*(js_consistency(o1,mb,rb)+js_consistency(o2,mb,rb))
            bal=.5*(balance_loss(o1['qf'])+balance_loss(o2['qf']))
            loss=cfg.lambda_instance*inst+cfg.lambda_consistency*cons+cfg.lambda_balance*bal
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step()
            losses.append([float(v.detach()) for v in (loss,inst,cons,bal)])
        model.eval()
        with torch.no_grad():
            vo=model([v[val] for v in normalized],mask_t[val],rel_t[val])
            # Label-free validation criterion, not test-set tuning.
            vloss=float((cfg.lambda_consistency*js_consistency(vo,mask_t[val],rel_t[val])+cfg.lambda_balance*balance_loss(vo['qf'])).item())
        row={'epoch':epoch+1,'loss':float(np.mean(losses,axis=0)[0]),'instance':float(np.mean(losses,axis=0)[1]),'consistency':float(np.mean(losses,axis=0)[2]),'balance':float(np.mean(losses,axis=0)[3]),'validation_unsupervised':vloss}
        history.append(row)
        if np.isfinite(vloss) and vloss<best-1e-6:
            best=vloss;best_epoch=epoch+1;best_state=copy.deepcopy(model.state_dict());wait=0
        else:
            wait+=1
            if wait>=cfg.patience: break
    if best_state is None: raise RuntimeError('No finite validation checkpoint')
    model.load_state_dict(best_state);model.eval()
    with torch.no_grad():
        embeddings=model([v for v in normalized],mask_t,rel_t)['fused'].numpy()
    km=KMeans(n_clusters=cfg.clusters,n_init=10,random_state=cfg.seed).fit(embeddings[train])
    # K-means fitted ONLY on train embeddings. No test-centroid fitting.
    pred_val=km.predict(embeddings[val]);pred_test=km.predict(embeddings[test]) if run_test else None
    def metrics(indices,pred):
        out={'n':len(indices),'cluster_counts':np.bincount(pred,minlength=cfg.clusters).tolist()}
        if len(np.unique(pred))>1 and len(indices)>len(np.unique(pred)):
            out['silhouette']=float(silhouette_score(embeddings[indices],pred))
        if labels is not None:
            y=np.asarray(labels)[indices]
            if len(y)!=len(pred):raise ValueError('Invalid labels')
            out['ARI']=float(adjusted_rand_score(y,pred));out['NMI']=float(normalized_mutual_info_score(y,pred))
        return out
    fingerprint=hashlib.sha256(b''.join([v.tobytes() for v in x]+[a.tobytes(),r.tobytes()])).hexdigest()
    run_id=fingerprint[:12]+'_'+str(cfg.seed)
    directory=Path(cfg.checkpoint_dir).resolve();directory.mkdir(parents=True,exist_ok=True)
    path=directory/('hamvcc_'+run_id+'.pt')
    if path.exists():raise FileExistsError('Checkpoint already exists; use a new output directory or seed')
    checkpoint={'model':best_state,'kmeans_centers':km.cluster_centers_,'normalization':stats,'config':asdict(cfg),'dims':[v.shape[1] for v in x], 'splits':{'train':train.tolist(),'validation':val.tolist(),'test':test.tolist()},'data_sha256':fingerprint,'best_epoch':best_epoch}
    torch.save(checkpoint,path)
    result={'architecture':'HAMV-CC common-space projected fusion, residual modulation, NT-Xent, weighted JS, cluster balance','loss_history':[v['loss'] for v in history],'history':history,'best_epoch':best_epoch,'checkpoint':str(path),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'validation':metrics(val,pred_val),'test':metrics(test,pred_test) if run_test else {'status':'TEST not executed (explicit authorization required)'},'manifest':{'kind':'new_exploratory_neural_run','manuscript_reproduction':False,'data_sha256':fingerprint,'config':asdict(cfg),'splits':{'train':train.tolist(),'validation':val.tolist(),'test':test.tolist()},'torch':torch.__version__,'numpy':np.__version__,'python':sys.version.split()[0],'platform':platform.platform(),'best_epoch':best_epoch,'test_executed':bool(run_test),'test_labels_for_training':False}}
    return result
