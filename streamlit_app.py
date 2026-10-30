"""Streamlit interface for the HAMV-CC research workbench. No Dash dependency."""
import io, json, hashlib, sys, platform, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core import load_csv, load_json, runs, gate, floor_strength, cluster_features, numeric_features
from diagnostics import cross_environment, gate_jacobian, coordinate_experiment
from document_pipeline import _extract_zip, cluster_documents, partition_assignable
from advanced_audit import stability_audit, cluster_keywords

st.set_page_config(page_title='HAMV-CC | Document Intelligence', page_icon='📚', layout='wide', initial_sidebar_state='expanded')
THEME_FILE = Path(__file__).resolve().parent / 'assets' / 'streamlit_theme.css'
if THEME_FILE.exists():
    st.markdown('<style>' + THEME_FILE.read_text(encoding='utf-8') + '</style>', unsafe_allow_html=True)

def page_header(kicker, heading, description):
    st.markdown(f'<div class="workspace-intro"><div class="section-kicker">{kicker}</div><div class="page-heading">{heading}</div><p class="page-description">{description}</p></div>', unsafe_allow_html=True)

def section_note(title, body):
    st.markdown(f'<div class="research-note"><strong>{title}</strong><span>{body}</span></div>', unsafe_allow_html=True)


def download_df(label, df, name):
    st.download_button(label,df.to_csv(index=False).encode('utf-8-sig'),file_name=name,mime='text/csv')
def json_download(label,obj,name):
    st.download_button(label,json.dumps(obj,indent=2,default=str).encode(),file_name=name,mime='application/json')
def safe_run(fn):
    try: return fn()
    except Exception as e:
        st.error(f'{type(e).__name__}: {e}')
        with st.expander('Technical details'):st.code(traceback.format_exc())
        return None

st.sidebar.markdown('<div class="sidebar-brand"><span class="brand-mark">H</span><div><strong>HAMV-CC</strong><small>DOCUMENT INTELLIGENCE LAB</small></div></div>',unsafe_allow_html=True)
st.sidebar.caption('Reliability-aware multi-view clustering')
page=st.sidebar.radio('WORKSPACES',['01 · Results','02 · Bounded gate lab','03 · Document features','04 · Diagnostics','05 · Archive upload & clustering'])
st.sidebar.divider()
st.sidebar.markdown('**RESEARCH SCOPE**')
st.sidebar.caption('Availability-masked reliability gating · geometric robustness · exploratory document grouping')
st.sidebar.info('Cluster geometry is not a validated semantic taxonomy. Recorded benchmarks and uploaded-document results remain separate.')

if page.startswith('01'):
    page_header('01 / BENCHMARKS', 'Published benchmark results', 'Explore recorded performance across experimental conditions and environments.')
    st.caption('Bundled Linux reference and Windows rerun records; no new benchmark execution.')
    env=st.selectbox('Recorded environment',['reference','windows'],format_func=lambda x: {'reference':'Linux reference','windows':'Windows rerun'}[x])
    df=runs(env)
    conditions=st.multiselect('Conditions',sorted(df.condition.unique()),default=sorted(df.condition.unique()))
    methods=st.multiselect('Methods',sorted(df.method.unique()),default=sorted(df.method.unique())[:8])
    metric=st.radio('Metric',['ARI','NMI'],horizontal=True)
    filt=df[df.condition.isin(conditions)&df.method.isin(methods)]
    if not filt.empty:
        summary=filt.groupby(['condition','method'],as_index=False)[metric].agg(['mean','std']).reset_index(drop=True)
        st.plotly_chart(px.bar(summary,x='condition',y='mean',color='method',barmode='group',error_y='std',title=f'{metric} across recorded runs'),use_container_width=True)
        st.dataframe(summary,use_container_width=True)
        download_df('Export filtered runs',filt,'hamvcc_recorded_runs.csv')
    else:st.warning('Select at least one condition and method.')

elif page.startswith('02'):
    page_header('02 / INTERACTIVE METHODS', 'Bounded reliability gate', 'Inspect how availability and quality scores determine each view’s participation.')
    st.caption('Manuscript Eq. (2): exact zero weights for missing views; floor bounds require a nonempty mask.')
    names=['Structural','Visual','Text']
    cols=st.columns(3); reliabilities=[];mask=[]
    for col,name in zip(cols,names):
        with col:
            mask.append(st.checkbox(f'{name} available',value=True,key=f'm_{name}'))
            reliabilities.append(st.slider(f'{name} reliability',0.,1.,.7,.01,key=f'r_{name}'))
    count=sum(mask)
    if count:
        rho=st.slider('Minimum available-view share ρ',0.01,round(1/count,3),min(.1,1/count),.01)
        kappa=floor_strength(rho,count)
        weights=gate(reliabilities,mask,kappa)
        a,b=st.columns(2)
        a.metric('Floor-constrained κ',f'{kappa:.4f}')
        b.metric('Minimum actual available share',f'{min(weights[np.array(mask,dtype=bool)]):.4f}')
        st.plotly_chart(px.bar(x=names,y=weights,labels={'x':'View','y':'Gate weight'},title='Normalized available-view weights'),use_container_width=True)
        _,jac=gate_jacobian(reliabilities,mask,kappa)
        st.plotly_chart(px.imshow(jac,x=names,y=names,aspect='auto',title='Local gate Jacobian ∂α/∂r'),use_container_width=True)
    else:st.error('All views missing: unassignable, not silently repaired.')

elif page.startswith('03'):
    page_header('03 / DOCUMENT ANALYSIS', 'Saved document-feature audit', 'Examine saved image features and their exploratory clustering geometry.')
    st.caption('Exploratory derived image features, not semantic validation.')
    df=load_csv('image_feature_matrix.csv')
    st.write(f'{len(df)} rows and {len(df.columns)} columns in the saved feature table.')
    st.dataframe(df.head(30),use_container_width=True)
    options=numeric_features(df)
    selected=st.multiselect('Numeric features',options,default=options[:min(18,len(options))])
    k=st.slider('Clusters K',2,min(12,max(2,len(df)-1)),2)
    seed=st.number_input('Seed',0,1000000,17)
    if st.button('Run feature clustering',type='primary'):
        def run():
            assignments,info=cluster_features(df,selected,k,seed)
            st.session_state['feature_result']=(assignments,info)
        safe_run(run)
    result=st.session_state.get('feature_result')
    if result is not None:
        assignments,info=result
        st.json(info)
        st.plotly_chart(px.scatter(assignments,x='PC1',y='PC2',color='cluster',hover_data=['document_id'],title='Feature clustering PCA projection'),use_container_width=True)
        st.dataframe(assignments,use_container_width=True)
        download_df('Export feature assignments',assignments,'feature_assignments.csv')
        json_download('Export feature diagnostics',info,'feature_diagnostics.json')

elif page.startswith('04'):
    page_header('04 / ROBUSTNESS', 'Numerical and geometric diagnostics', 'Inspect cross-environment differences and coordinate-transformation behavior.')
    metric=st.selectbox('Comparison metric',['ARI','NMI'])
    tol=st.select_slider('Numerical tolerance',options=[1e-12,1e-10,1e-8,1e-6,1e-4],value=1e-10)
    if st.button('Compare environments'):
        safe_run(lambda: st.session_state.update({'cross_env':cross_environment(metric,tol)}))
    result=st.session_state.get('cross_env')
    if result is not None:
        if isinstance(result,pd.DataFrame):
            st.dataframe(result,use_container_width=True)
            download_df('Export comparison',result,'cross_environment.csv')
        else:st.write(result)
    st.subheader('Coordinate-transformation experiment')
    seed=st.number_input('Simulation seed',0,1000000,42,key='geo_seed')
    if st.button('Run coordinate robustness simulation'):
        safe_run(lambda: st.session_state.update({'geometry':coordinate_experiment(seed=int(seed))}))
    if 'geometry' in st.session_state:
        g=st.session_state['geometry']
        if isinstance(g,pd.DataFrame):st.dataframe(g,use_container_width=True)
        else:st.write(g)
    st.caption('Simulation output is a new diagnostic, not part of the manuscript’s fixed experimental record.')

else:
    page_header('05 / DOCUMENT WORKSPACE', 'Upload, select K, and cluster documents', 'Import mixed document archives, inspect extraction, select K, and review every cluster assignment.')
    st.caption('Upload images, text documents, PDFs, or ZIP archives. This exploratory pipeline uses structural, visual, and OCR/text views; it is not a trained semantic classifier.')
    section_note('Direct archive upload', 'Choose a ZIP from your computer. No local file path or additional upload service is required.')
    st.markdown('### Import your document archive')
    file=st.file_uploader('Upload ZIP archive',type=['zip'],accept_multiple_files=False,key='archive_zip')
    individual=st.file_uploader('Or select multiple individual documents',type=['png','jpg','jpeg','tif','tiff','bmp','webp','txt','md','pdf'],accept_multiple_files=True,key='individual_documents')
    st.caption('Limits: 500 MB ZIP, 1,500 archive entries, 180 MB uncompressed, 12 MB per item. Uploaded data is processed in the local Streamlit Python process.')
    if file is not None or individual:
        if file is not None: st.write(f'Selected: **{file.name}** — {len(file.getvalue())/1048576:.2f} MiB')
        st.success('Files selected. Click Import to extract and inspect documents.')
        if st.button('Import and inspect archive',type='primary'):
            def ingest():
                if file is not None:
                    data=file.getvalue()
                else:
                    import zipfile
                    buff=io.BytesIO()
                    with zipfile.ZipFile(buff,'w',compression=zipfile.ZIP_DEFLATED) as z:
                        for item in individual:
                            z.writestr(Path(item.name).name,item.getvalue())
                    data=buff.getvalue()
                if len(data)>500_000_000:raise ValueError('ZIP exceeds 500 MB limit')
                records,skipped=_extract_zip(io.BytesIO(data))
                st.session_state['archive_records']=records
                st.session_state['archive_skipped']=skipped
                st.session_state['archive_digest']=hashlib.sha256(data).hexdigest()
                st.session_state.pop('archive_cluster',None)
            with st.spinner('Extracting documents and OCR (if installed)…'):
                safe_run(ingest)
    records=st.session_state.get('archive_records')
    if records:
        all_records=records
        records, unassignable=partition_assignable(all_records)
        st.markdown('### Archive overview')
        skipped=st.session_state.get('archive_skipped',[])
        a,b,c=st.columns(3)
        a.metric('Documents eligible for clustering',len(records))
        b.metric('Skipped/duplicate files',len(skipped))
        c.metric('Unassignable documents',len(unassignable))
        if unassignable:
            st.warning(f'{len(unassignable)} readable document(s) have no usable views and cannot be assigned to clusters. They are excluded explicitly, not silently repaired.')
            with st.expander('Unassignable documents — inspect and export',expanded=True):
                st.dataframe(pd.DataFrame(unassignable),use_container_width=True)
                download_df('Download unassignable documents (CSV)',pd.DataFrame(unassignable),'hamvcc_unassignable_documents.csv')
        if skipped:
            with st.expander('Skipped files and reasons'):
                st.dataframe(pd.DataFrame(skipped),use_container_width=True)
        kmax=min(20,len(records)-1)
        if len(records) < 3:
            st.warning('At least three unique documents are needed to evaluate silhouette scores. Upload a larger archive.')
            st.stop()
        st.markdown('### Clustering configuration')
        c1,c2,c3,c4=st.columns(4)
        with c1:kmin=st.number_input('Minimum K',min_value=2,max_value=kmax,value=2)
        with c2:kend=st.number_input('Maximum K',min_value=kmin,max_value=kmax,value=min(12,kmax))
        with c3:seed=st.number_input('Random seed',0,1000000,17,key='archive_seed')
        with c4:floor=st.number_input('Minimum view share ρ',min_value=.01,max_value=.32,value=.1,step=.01)
        st.caption('K is selected by maximum silhouette. The elbow curve is a supporting diagnostic, not a unique ground-truth K.')
        if st.button('Find best K and cluster documents',type='primary'):
            with st.spinner('Computing multi-view representations and silhouette / elbow diagnostics…'):
                safe_run(lambda:st.session_state.update({'archive_cluster':cluster_documents(records,kmin,kend,seed,floor)}))
        result=st.session_state.get('archive_cluster')
        if result:
            st.success(f"Silhouette-selected K = {result['best_k']} (exploratory)")
            diagnostics=pd.DataFrame(result['diagnostics'])
            c1,c2=st.columns(2)
            with c1:st.plotly_chart(px.line(diagnostics,x='K',y='silhouette',markers=True,title='Silhouette versus K'),use_container_width=True)
            with c2:st.plotly_chart(px.line(diagnostics,x='K',y='inertia',markers=True,title='Elbow curve (inertia versus K)'),use_container_width=True)
            st.subheader('Cluster quality and stability')
            x=np.asarray(result['embedding'])
            st.caption('Seed stability measures partition reproducibility; it does not establish correct semantic categories.')
            if st.button('Run multi-seed stability audit'):
                safe_run(lambda:st.session_state.update({'stability_audit':stability_audit(x,result['best_k'],8)}))
            if 'stability_audit' in st.session_state: st.json(st.session_state['stability_audit'])
            st.subheader('Cluster visualization')
            from sklearn.decomposition import PCA
            projection=PCA(n_components=2,random_state=17).fit_transform(x)
            view_df=pd.DataFrame({'PC1':projection[:,0],'PC2':projection[:,1], 'cluster':[str(d['cluster']) for d in result['documents']], 'filename':[d['filename'] for d in result['documents']]})
            st.plotly_chart(px.scatter(view_df,x='PC1',y='PC2',color='cluster',hover_data=['filename'],title='Exploratory two-dimensional PCA projection'),use_container_width=True)
            st.caption('PCA is only a visualization; clustering is performed on the full fused feature matrix.')
            st.subheader('Indicative cluster keywords')
            keywords=cluster_keywords(records,[d['cluster'] for d in result['documents']])
            if keywords:st.dataframe(pd.DataFrame(keywords),use_container_width=True,hide_index=True)
            else:st.info('No sufficient text was extracted for keyword summaries.')
            st.subheader('Cluster summary')
            st.caption('Cluster sizes summarize grouping, not independently verified document categories.')
            st.dataframe(pd.DataFrame(result['summary']),use_container_width=True)
            st.subheader('Document directory & cluster assignments')
            st.caption('Filter by cluster or search filenames and extracted text. Export all assignments below.')
            detail=pd.DataFrame(result['documents'])
            cluster=st.selectbox('Filter by cluster',['All']+sorted(detail.cluster.unique().tolist(),key=int))
            filtered=detail if cluster=='All' else detail[detail.cluster==cluster]
            query=st.text_input('Search filename or extracted text')
            if query:filtered=filtered[filtered.filename.str.contains(query,case=False,regex=False)|filtered.text_preview.str.contains(query,case=False,regex=False)]
            st.dataframe(filtered,use_container_width=True,hide_index=True)
            st.subheader('Individual document inspector')
            selected_doc=st.selectbox('Select document',options=filtered.document_id.tolist(),format_func=lambda d: next((r['filename'] for r in records if r['document_id']==d),d)) if len(filtered) else None
            if selected_doc:
                rec=next(r for r in records if r['document_id']==selected_doc)
                detail_row=detail.loc[detail.document_id==selected_doc].iloc[0]
                left,right=st.columns([1,2])
                with left:
                    st.metric('Assigned cluster',int(detail_row['cluster']))
                    st.write('**File:**',rec['filename'])
                    st.write('**SHA-256:**',rec['sha256'])
                    st.write('**OCR status:**',rec['ocr_status'])
                    st.write('**Views:**',{'structural':bool(detail_row['has_image']),'visual':bool(detail_row['has_image']),'text':bool(detail_row['has_text'])})
                with right:
                    st.write('**Extracted text preview**')
                    st.text_area('Extracted text',rec['text'][:10000] or '(No text extracted)',height=220,disabled=True,label_visibility='collapsed')
                    st.write('**Reliability proxy weights**')
                    st.bar_chart(pd.DataFrame({'weight':[detail_row['structural_weight'],detail_row['visual_weight'],detail_row['text_weight']]},index=['Structural','Visual','Text']))
            download_df('Download document assignments (CSV)',detail,'hamvcc_document_assignments.csv')
            download_df('Download K diagnostics (CSV)',diagnostics,'hamvcc_k_diagnostics.csv')
            json_download('Download reproducibility report (JSON)',{'archive_sha256':st.session_state.get('archive_digest'),'best_k':result['best_k'],'protocol':result['protocol'],'diagnostics':result['diagnostics'],'summary':result['summary'],'keywords':keywords,'stability':st.session_state.get('stability_audit'),'skipped':skipped,'unassignable':unassignable,'python':sys.version,'platform':platform.platform()},'hamvcc_archive_manifest.json')
            st.warning('K is selected on the uploaded corpus; the silhouette and elbow are internal geometry diagnostics, not held-out accuracy or validated semantic document categories.')
    else:st.info('Upload a ZIP to display document counts, K selection controls, and cluster assignments.')
