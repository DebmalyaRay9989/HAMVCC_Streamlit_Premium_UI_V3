import io,zipfile
from document_pipeline import _extract_zip,cluster_documents
from advanced_audit import stability_audit,cluster_keywords

def test_workflow():
    f=io.BytesIO()
    with zipfile.ZipFile(f,'w') as z:
        for i in range(12):z.writestr(f'file_{i}.txt',('sports baseball match score '*10 if i<6 else 'space science satellite orbit '*10)+str(i))
    f.seek(0)
    records,skipped=_extract_zip(f)
    assert len(records)==12
    result=cluster_documents(records,2,4,17,.1)
    assert 2<=result['best_k']<=4
    assert len(result['documents'])==12
    audit=stability_audit(result['embedding'],result['best_k'],4)
    assert 0<=audit['mean_pairwise_ari']<=1
    assert cluster_keywords(records,[d['cluster'] for d in result['documents']])
if __name__=='__main__':
    test_workflow();print('PASS: ingestion, clustering, stability, keyword audit')
