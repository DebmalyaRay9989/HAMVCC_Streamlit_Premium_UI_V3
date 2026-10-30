import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from document_pipeline import partition_assignable, cluster_documents

def test_partition():
    items=[{'filename':'blank.pdf','document_id':'a','type':'.pdf','text':'','image_features':None,'ocr_status':'pdf_text_extraction_only'}, {'filename':'notes.txt','document_id':'b','type':'.txt','text':'hello','image_features':None,'ocr_status':'not_applicable'}]
    usable,excluded=partition_assignable(items)
    assert len(usable)==1 and len(excluded)==1
    assert excluded[0]['assignment_status']=='UNASSIGNABLE'
    assert excluded[0]['cluster'] is None

if __name__=='__main__':
    test_partition()
    print('PASS: all-missing records explicitly excluded')
