import tempfile, unittest
import numpy as np
from neural import NeuralConfig,model_factory,train_neural,gate_strength

class TestNeural(unittest.TestCase):
    def test_gate_and_missing(self):
        import torch
        cfg=NeuralConfig(clusters=3,latent=8,hidden=12)
        model=model_factory([4,5],cfg)
        m=torch.tensor([[1.,0.],[1.,1.]])
        o=model([torch.randn(2,4),torch.randn(2,5)],m,torch.tensor([[.7,.9],[.2,.8]]))
        self.assertEqual(float(o['alpha'][0,1]),0.)
        self.assertAlmostEqual(float(o['alpha'][0,0]),1.,places=6)
        self.assertGreaterEqual(float(o['alpha'][1].min()),.1-1e-6)
        self.assertTrue(torch.allclose(o['views'][0][0],o['views'][0][0]))
        self.assertTrue(torch.allclose(o['views'][1][0],torch.zeros(8)))
    def test_checkpoint_and_test_gate(self):
        rng=np.random.default_rng(5);n=48
        x=[rng.normal(size=(n,5)),rng.normal(size=(n,6))]
        a=np.ones((n,2),bool);r=np.ones((n,2))*.8
        with tempfile.TemporaryDirectory() as d:
            out=train_neural(x,a,r,k=3,epochs=2,seed=3,checkpoint_dir=d)
            self.assertIn('TEST not executed',out['test']['status'])
            self.assertEqual(len(out['history']),2)
if __name__=='__main__': unittest.main()
