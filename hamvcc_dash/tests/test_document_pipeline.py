import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from document_pipeline import cluster_documents
class DocumentPipelineTest(unittest.TestCase):
    def test_text_only(self):
        docs=[{'document_id':str(i),'filename':f'{i}.txt','type':'.txt','size_bytes':10,'sha256':str(i),'text':('cats pets whiskers ' if i<5 else 'rockets orbit space ')*5+str(i),'image_features':None,'ocr_status':'not_applicable'} for i in range(10)]
        result=cluster_documents(docs,2,4)
        self.assertEqual(len(result['documents']),10)
        self.assertTrue(all(d['structural_weight']==0 for d in result['documents']))
        self.assertEqual(sum(x['document_count'] for x in result['summary']),10)
    def test_invalid_k(self):
        with self.assertRaises(ValueError):cluster_documents([],2,5)
if __name__=='__main__':unittest.main()
