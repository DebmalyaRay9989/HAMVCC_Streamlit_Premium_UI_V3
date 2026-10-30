import base64
import hashlib
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from core import gate, floor_strength, load_json, load_csv, runs, DATA, parse_upload, cluster_features, numeric_features
from app import app, update_results, update_docs, update_gate, export_clusters

class ScientificChecks(unittest.TestCase):
    def test_gate_invariants(self):
        for count in [1,2,3]:
            mask=np.arange(3)<count
            for k in [0,.752,4,1000]:
                w=gate([.2,.7,1],mask,k)
                self.assertAlmostEqual(w.sum(),1)
                self.assertTrue((w[~mask]==0).all())
        np.testing.assert_allclose(gate([1,0,.5],[1,1,0],0),[.5,.5,0])
        with self.assertRaises(ValueError):gate([.5]*3,[0]*3,1)
        with self.assertRaises(ValueError):gate([np.nan,0,1],[1,1,1],1)
        with self.assertRaises(ValueError):gate([.5]*3,[1,2,0],1)
    def test_floor_and_sensitivity(self):
        k=floor_strength(.1,3)
        self.assertAlmostEqual(k,.7520386983881371)
        self.assertAlmostEqual(gate([0,1,1],[1]*3,k)[0],.1)
        rng=np.random.default_rng(42)
        for _ in range(100):
            r=rng.random(3);r2=np.clip(r+rng.uniform(-.1,.1,3),0,1)
            self.assertLessEqual(np.abs(gate(r,[1]*3,k)-gate(r2,[1]*3,k)).sum(),2*k*np.max(np.abs(r-r2))+1e-12)
    def test_provenance_and_matched_runs(self):
        for item in load_json('provenance.json'):
            self.assertEqual(hashlib.sha256((DATA/item['file']).read_bytes()).hexdigest(),item['sha256'])
        for env,analysis in [('reference','reference_analysis_v2.json'),('windows','windows_analysis.json')]:
            df=runs(env)
            self.assertFalse(df.duplicated(['condition','split','seed','method']).any())
            self.assertEqual(len(df[['condition','split','seed']].drop_duplicates()),160)
            for cond,cs in load_json(analysis)['contrasts'].items():
                p=df[df.condition==cond].pivot(index=['split','seed'],columns='method',values='ARI')
                effect=(p.learned_rho-p.avail_only).groupby('split').mean()
                np.testing.assert_allclose(effect,cs['learned_rho-avail_only']['split_effects'],atol=1e-12)
        a=runs('reference').set_index(['condition','split','seed','method'])[['ARI','NMI']]
        b=runs('windows').set_index(['condition','split','seed','method'])[['ARI','NMI']]
        self.assertEqual(int(((a-b).abs()>1e-10).sum().sum()),6806)
    def test_csv_validation_and_clustering(self):
        df=load_csv('image_feature_matrix.csv')
        result,info=cluster_features(df,numeric_features(df),2,42)
        self.assertEqual(len(result),199)
        self.assertAlmostEqual(info['silhouette'],.2622665950041013,places=8)
        with self.assertRaises(ValueError):cluster_features(df,[],2,42)
        with self.assertRaises(ValueError):cluster_features(df,['aspect_ratio'],2,42)
        with self.assertRaises(ValueError):cluster_features(df,numeric_features(df),None,42)
        text='document_id,x,y\na,1,3\nb,2,4\nc,3,2\nd,4,1\n'
        encoded='data:text/csv;base64,'+base64.b64encode(text.encode()).decode()
        self.assertEqual(len(parse_upload(encoded,'features.csv')),4)
        with self.assertRaises(ValueError):parse_upload(encoded,'bad.exe')
    def test_callbacks_and_routes(self):
        client=app.server.test_client()
        for route in ['/','/_dash-layout','/_dash-dependencies','/assets/style.css']:
            with client.get(route) as response:
                self.assertEqual(response.status_code,200)
        for env in ['reference','windows']:
            for condition in ['M0+D50','M20+D50','M40+D50','M20+clean']:
                for metric in ['ARI','NMI']:
                    outputs=update_results(env,condition,metric,['avail_only','learned_rho'])
                    self.assertEqual(len(outputs[0].data),1)
        self.assertEqual(len(update_results('reference','M0+D50','ARI',[])[2]),0)
        for method in ['kmeans','gaussian_mixture','agglomerative','spectral']:
            for k in [2,3]:update_docs(method,k,'silhouette')
        update_gate([],.8,.6,.3,'floor',.1,1,.1)
        for mask in [['Text'],['Text','Visual'],['Structural','Visual','Text']]:
            update_gate(mask,.8,.6,.3,'manual',.1,4,.1)

if __name__=='__main__':unittest.main(verbosity=2)
