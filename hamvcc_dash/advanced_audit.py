"""Independent exploratory diagnostics; never claim semantic accuracy without labels."""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

def stability_audit(embedding, k, seeds=8):
    x=np.asarray(embedding,float)
    if len(x)<4 or k>=len(x): raise ValueError('Not enough rows for stability audit')
    labels=[KMeans(n_clusters=int(k),n_init=10,random_state=s).fit_predict(x) for s in range(int(seeds))]
    pairs=[adjusted_rand_score(labels[i],labels[j]) for i in range(len(labels)) for j in range(i+1,len(labels))]
    return {'mean_pairwise_ari':float(np.mean(pairs)), 'min_pairwise_ari':float(np.min(pairs)), 'max_pairwise_ari':float(np.max(pairs)), 'seed_count':len(labels), 'pair_count':len(pairs), 'note':'Pairwise seed comparisons are dependent; ARI is partition agreement, not semantic accuracy.'}

def cluster_keywords(records, assignments, top_n=8):
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts=[r.get('text','') for r in records]
    if sum(bool(t.strip()) for t in texts)<2:return []
    try:
        tf=TfidfVectorizer(stop_words='english',max_features=8000,ngram_range=(1,2),min_df=1).fit_transform(texts)
        vocab=TfidfVectorizer
        # Refit once to retain vocabulary and matrix together.
        vec=TfidfVectorizer(stop_words='english',max_features=8000,ngram_range=(1,2),min_df=1)
        tf=vec.fit_transform(texts)
        terms=vec.get_feature_names_out()
        rows=[]
        for cluster in sorted(set(assignments)):
            ix=np.flatnonzero(np.asarray(assignments)==cluster)
            mean=np.asarray(tf[ix].mean(axis=0)).ravel()
            terms_ix=np.argsort(-mean)[:top_n]
            rows.append({'cluster':int(cluster),'document_count':int(len(ix)),'keywords':', '.join(terms[j] for j in terms_ix if mean[j]>0)})
        return rows
    except ValueError:return []
