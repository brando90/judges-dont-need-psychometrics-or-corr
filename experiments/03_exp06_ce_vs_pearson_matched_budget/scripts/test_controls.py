"""Deterministic numerical tests; no empirical data are opened."""
import unittest
import numpy as np
from ce_mixture import fit, predict_mixture


class Controls(unittest.TestCase):
    def test_sparse_inputs_certified(self):
        rng=np.random.default_rng(711)
        for _ in range(100):
            q=rng.multinomial(3,np.ones(6)/6,size=(60,5))/3
            q=q*(1-6e-6)+1e-6
            p=rng.multinomial(3,np.ones(6)/6,size=60)/3
            _,receipt=fit(q,p,'ce')
            self.assertTrue(receipt['success'],receipt)

    def test_recovery(self):
        rng = np.random.default_rng(20261109)
        q = rng.dirichlet(np.ones(6), size=(30,5))
        truth = np.array([.1,.2,.3,.15,.25])
        p = predict_mixture(q, truth)
        for objective in ['ce','pearson']:
            w, receipt = fit(q,p,objective)
            self.assertTrue(receipt['success'],receipt)
            np.testing.assert_allclose(predict_mixture(q,w),p,atol=1e-6)

    def test_two_candidate_grid(self):
        rng = np.random.default_rng(20261110)
        for _ in range(8):
            q = rng.dirichlet(np.ones(6),size=(20,2))
            p = rng.dirichlet(np.ones(6),size=20)
            weights = np.column_stack([np.linspace(0,1,10001),np.linspace(1,0,10001)])
            forecasts = np.einsum('njr,aj->anr',q,weights)
            ce = -np.mean(np.sum(p[None]*np.log(forecasts),axis=2),axis=1)
            w, receipt = fit(q,p,'ce')
            self.assertTrue(receipt['success'],receipt)
            loss = -np.mean(np.sum(p*np.log(predict_mixture(q,w)),axis=1))
            self.assertLessEqual(loss,ce.min()+1e-10)

    def test_duplicates_and_constants(self):
        rng = np.random.default_rng(44)
        p = rng.dirichlet(np.ones(6),size=20)
        q = np.stack([p,p],axis=1)
        for objective in ['ce','pearson']:
            w,receipt = fit(q,p,objective)
            self.assertTrue(receipt['success'])
            np.testing.assert_allclose(predict_mixture(q,w),p,atol=1e-10)
        _,receipt = fit(np.full((20,3,6),1/6),p,'pearson')
        self.assertTrue(receipt['undefined_correlation'])

    def test_ce_gradient(self):
        rng = np.random.default_rng(45)
        q = rng.dirichlet(np.ones(6),size=(20,5))
        p = rng.dirichlet(np.ones(6),size=20)
        w = np.full(5,.2)
        pred = predict_mixture(q,w)
        g = -np.einsum('nr,njr->j',p/pred,q)/len(p)
        d = np.array([1.,-1.,0,0,0])
        f = lambda v:-np.mean(np.sum(p*np.log(predict_mixture(q,v)),axis=1))
        self.assertAlmostEqual((f(w+1e-6*d)-f(w-1e-6*d))/2e-6,g@d,places=8)


if __name__ == '__main__':
    unittest.main()
