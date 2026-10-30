"""Document ingestion and exploratory HAMV-CC-inspired clustering.
No semantic claims without independent labels; K selected without test labels.
"""
import base64, hashlib, io, json, re, zipfile
from pathlib import PurePosixPath
import numpy as np
import pandas as pd

MAX_ARCHIVE=45_000_000
MAX_FILES=1500
MAX_ITEM=12_000_000
ALLOWED={'.png','.jpg','.jpeg','.tif','.tiff','.bmp','.webp','.txt','.md','.pdf'}

def extract_archive(contents, filename):
    if not filename or not filename.lower().endswith('.zip'): raise ValueError('Upload a ZIP of image, text, or PDF documents.')
    raw=base64.b64decode(contents.partition(',')[2],validate=True)
    if len(raw)>MAX_ARCHIVE: raise ValueError('Archive exceeds 45 MB upload limit.')
    return _extract_zip(io.BytesIO(raw))

def extract_archive_file(path):
    """Load a local ZIP without sending its bytes through the browser."""
    from pathlib import Path
    path=Path(path)
    if path.suffix.lower()!='.zip':raise ValueError('Select a .zip file.')
    if path.stat().st_size>500_000_000:raise ValueError('ZIP exceeds the 500 MB local-file limit.')
    return _extract_zip(path)

def _extract_zip(source):
    records=[]; seen={}; skipped=[]
    with zipfile.ZipFile(source) as z:
        files=[i for i in z.infolist() if not i.is_dir() and not i.filename.startswith('__MACOSX/')]
        if len(files)>MAX_FILES: raise ValueError('Maximum 1500 files per archive.')
        if sum(i.file_size for i in files)>180_000_000: raise ValueError('Uncompressed archive exceeds 180 MB.')
        for info in files:
            path=PurePosixPath(info.filename)
            if '..' in path.parts or info.file_size>MAX_ITEM or info.flag_bits & 1:
                skipped.append({'file':info.filename,'reason':'unsafe, encrypted, or oversized'});continue
            suffix=path.suffix.lower()
            if suffix not in ALLOWED: continue
            blob=z.read(info)
            digest=hashlib.sha256(blob).hexdigest()
            if digest in seen:
                skipped.append({'file':info.filename,'reason':'exact duplicate of '+seen[digest]});continue
            seen[digest]=info.filename
            rec={'document_id':digest[:16],'filename':info.filename,'type':suffix,'size_bytes':len(blob),'sha256':digest,'text':'','image_features':None,'ocr_status':'not_applicable'}
            try:
                if suffix in ('.txt','.md'):
                    rec['text']=blob.decode('utf-8-sig',errors='replace')[:100000]
                elif suffix=='.pdf':
                    try:
                        from pypdf import PdfReader
                        reader=PdfReader(io.BytesIO(blob));rec['text']='\n'.join(p.extract_text() or '' for p in list(reader.pages)[:30])[:100000]
                        rec['ocr_status']='pdf_text_extraction_only'
                    except ImportError: rec['ocr_status']='pypdf_not_installed'
                else:
                    from PIL import Image,ImageOps,ImageStat,ImageFilter
                    im=Image.open(io.BytesIO(blob));im.load()
                    if im.width*im.height>32_000_000: raise ValueError('image pixel limit exceeded')
                    gray=ImageOps.grayscale(im)
                    thumb=ImageOps.pad(gray,(128,128),color=255)
                    a=np.asarray(thumb,dtype=np.float32)/255.
                    ink=1-a
                    row=ink.mean(axis=1).reshape(16,8).mean(axis=1)
                    col=ink.mean(axis=0).reshape(16,8).mean(axis=1)
                    gx=np.abs(np.diff(a,axis=1));gy=np.abs(np.diff(a,axis=0))
                    rec['image_features']=np.r_[row,col,[ink.mean(),ink.std(),np.quantile(ink,.9),gx.mean(),gy.mean(),np.mean(ink>.5),np.log1p(im.width),np.log1p(im.height),im.width/max(im.height,1)]].astype(float).tolist()
                    rec['width']=im.width;rec['height']=im.height
                    try:
                        import pytesseract
                        rec['text']=pytesseract.image_to_string(im,config='--psm 3',timeout=8)[:100000]
                        rec['ocr_status']='completed'
                    except ImportError: rec['ocr_status']='pytesseract_not_installed'
                    except Exception as e: rec['ocr_status']='ocr_unavailable_or_failed'
            except Exception as e:
                skipped.append({'file':info.filename,'reason':str(e)[:140]});continue
            records.append(rec)
    if len(records)<4: raise ValueError('At least four unique readable documents are required.')
    return records,skipped

def cluster_documents(records,k_min=2,k_max=12,seed=17,floor=.1):
    from sklearn.preprocessing import StandardScaler,normalize
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score,davies_bouldin_score
    from threadpoolctl import threadpool_limits
    n=len(records);k_min=int(k_min);k_max=int(k_max)
    if not 2<=k_min<=k_max< n:raise ValueError('K must satisfy 2 <= Kmin <= Kmax < number of documents.')
    img=np.array([r['image_features'] is not None for r in records],bool)
    txt=np.array([bool(r['text'].strip()) for r in records],bool)
    # Image structural and visual feature groups are separate blocks; text is optional.
    mask=np.column_stack([img,img,txt]); M=3
    if not mask.any(axis=1).all():
        missing=[r['filename'] for r,valid in zip(records,mask.any(axis=1)) if not valid]
        raise ValueError(f'{len(missing)} documents have no usable structural, visual, or text view. Exclude these unassignable documents before clustering. Examples: {missing[:5]}')
    raw=np.array([r['image_features'] if r['image_features'] is not None else [0.]*41 for r in records],float)
    if raw.shape[1]!=41:raise ValueError('Image feature schema mismatch.')
    # Separate structural and appearance proxies from the image; no pretrained semantic visual model.
    structural=np.c_[raw[:,:32],raw[:,35:]]
    visual=raw[:,32:35]
    def transform(a,exists):
        out=np.zeros((n,a.shape[1]));
        if exists.sum()>1:out[exists]=StandardScaler().fit_transform(a[exists])
        return normalize(out)
    S=transform(structural,img);V=transform(visual,img)
    text_rows=[r['text'] if txt[i] else '' for i,r in enumerate(records)]
    T=np.zeros((n,16),float)
    if txt.sum()>1:
        try:
            tf=TfidfVectorizer(max_features=4000,ngram_range=(1,2),min_df=1,stop_words='english').fit_transform(text_rows)
            if tf.shape[1]>1:
                dim=min(16,tf.shape[1]-1,n-1)
                projected=TruncatedSVD(n_components=dim,random_state=seed).fit_transform(tf)
                T[:,:dim]=projected
            else:T[:,0]=np.asarray(tf.sum(axis=1)).ravel()
            T=normalize(T)
        except ValueError: pass
    # Declared transparent proxies, NOT learned/calibrated semantic reliability.
    r=np.column_stack([np.clip(1-np.abs(raw[:,33]-.25),0,1),np.clip(1-raw[:,34],0,1),np.clip(np.array([len(re.findall(r'\w+',t)) for t in text_rows])/100,0,1)])
    r[~mask]=0
    if not 0<floor<1/3:raise ValueError('floor must lie between 0 and 1/3.')
    kappa=.5*np.log((1/floor-1)/2)
    logits=kappa*(2*r-1);logits[~mask]=-np.inf
    shifted=logits-np.max(logits,axis=1,keepdims=True)
    weights=np.exp(shifted);weights/=weights.sum(axis=1,keepdims=True)
    # Manuscript's block concatenation: sqrt(M_i) alpha_m e_m, no alignment assumption.
    block=np.concatenate([np.sqrt(mask.sum(axis=1))[:,None]*weights[:,j:j+1]*x for j,x in enumerate([S,V,T])],axis=1)
    masked=np.concatenate([np.sqrt(mask.sum(axis=1))[:,None]*mask[:,j:j+1]*x/mask.sum(axis=1,keepdims=True) for j,x in enumerate([S,V,T])],axis=1)
    diagnostics=[]; labels_by_k={}
    with threadpool_limits(limits=1):
        for k in range(k_min,k_max+1):
            model=KMeans(n_clusters=k,n_init=10,random_state=seed)
            lab=model.fit_predict(block)
            sil=float(silhouette_score(block,lab)) if len(set(lab))>1 else None
            diagnostics.append({'K':k,'silhouette':sil,'inertia':float(model.inertia_),'davies_bouldin':float(davies_bouldin_score(block,lab)) if len(set(lab))>1 else None})
            labels_by_k[k]=lab
    valid=[d for d in diagnostics if d['silhouette'] is not None]
    if not valid:raise ValueError('No K produced at least two distinct clusters.')
    best=max(valid,key=lambda d:(d['silhouette'],-d['K']))['K']
    labels=labels_by_k[best]
    # Elbow is diagnostic, not automatically a unique optimum.
    report=[]
    for i,rec in enumerate(records):
        report.append({'document_id':rec['document_id'],'filename':rec['filename'],'file_type':rec['type'],'size_bytes':rec['size_bytes'],'sha256':rec['sha256'],'cluster':int(labels[i]),'has_image':bool(img[i]),'has_text':bool(txt[i]),'ocr_status':rec['ocr_status'],'text_preview':re.sub(r'\s+',' ',rec['text'])[:180], 'structural_reliability_proxy':round(float(r[i,0]),4),'visual_reliability_proxy':round(float(r[i,1]),4),'text_reliability_proxy':round(float(r[i,2]),4),'structural_weight':round(float(weights[i,0]),4),'visual_weight':round(float(weights[i,1]),4),'text_weight':round(float(weights[i,2]),4)})
    return {'best_k':int(best),'diagnostics':diagnostics,'documents':report,'summary':pd.DataFrame(report).groupby('cluster').agg(document_count=('document_id','count'),image_count=('has_image','sum'),text_count=('has_text','sum')).reset_index().to_dict('records'),'embedding':block.tolist(),'masked_embedding':masked.tolist(),'protocol':{'status':'exploratory_unsupervised','selection':'maximum silhouette on uploaded corpus; elbow shown for visual inspection','fusion':'bounded proxy-gated block concatenation','kappa':float(kappa),'minimum_share_floor':floor,'seed':seed,'independent_test':'not performed','semantic_validation':'not available without independently labelled documents','limitations':'Image-derived features and TF-IDF are lightweight proxies, not the full trained HAMV-CC neural pipeline. Uploaded corpus is used for K selection and fitting; reported silhouette is descriptive, not held-out performance.'}}


def partition_assignable(records):
    """Fail closed on all-missing documents; preserve an explicit exclusion audit."""
    assignable, excluded = [], []
    for rec in records:
        has_image = rec.get('image_features') is not None
        has_text = bool((rec.get('text') or '').strip())
        if has_image or has_text:
            assignable.append(rec)
        else:
            excluded.append({
                'document_id': rec.get('document_id'),
                'filename': rec.get('filename'),
                'file_type': rec.get('type'),
                'ocr_status': rec.get('ocr_status'),
                'reason': 'No usable view: neither image features nor extracted text',
                'cluster': None,
                'assignment_status': 'UNASSIGNABLE',
            })
    return assignable, excluded
