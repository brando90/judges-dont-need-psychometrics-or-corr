"""Analytic, sampler/scorer and split-integrity checks (no measured judge calls)."""
import json
from pathlib import Path
import unittest
import numpy as np
from scipy.stats import binom
import run_study as s

CONFIG=json.loads((Path(__file__).parent/'protocol.json').read_text())


class StudyTests(unittest.TestCase):
    def test_binomial_against_scipy(self):
        t=np.array([.1,.2,.5,.8,.9])
        np.testing.assert_allclose(s.population(t),binom.pmf(np.arange(6),5,t[:,None]),atol=1e-15)
        np.testing.assert_allclose(s.population(t).sum(axis=1),1)

    def test_exact_shared_mean_and_strict_proper_losses(self):
        t=np.linspace(.1,.9,17)
        p=s.population(t)
        q,mu=s.candidates(t,np.array([[1,2,3]]),CONFIG)
        np.testing.assert_allclose(q[:5]@s.SCORES,np.tile(5*t,(5,1)),atol=1e-14)
        base=s.expected_metrics(p,q[0],mu[0])
        for i in range(1,5):
            row=s.expected_metrics(p,q[i],mu[i])
            self.assertAlmostEqual(row['ce']-base['ce'],row['kl'])
            self.assertAlmostEqual(row['brier']-base['brier'],np.mean(np.sum((q[i]-p)**2,axis=1)))
            self.assertAlmostEqual(row['rps']-base['rps'],np.mean((np.cumsum(q[i]-p,axis=1)[:,:-1])**2))
            for key in ('ce','brier','rps'):
                self.assertGreater(row[key],base[key])
            self.assertEqual(row['nmse'],base['nmse'])
            self.assertEqual(row['pearson_conditional_mean'],base['pearson_conditional_mean'])

    def test_permutation_keeps_only_marginal(self):
        t=np.array([r['t'] for r in s.items(CONFIG) if r['split']=='test'])
        p,q=s.population(t),s.population(1-t)
        np.testing.assert_allclose(p.mean(axis=0),q.mean(axis=0),atol=1e-15)
        self.assertGreater(s.expected_metrics(p,q,5*(1-t))['kl'],0)

    def test_empirical_scoring_equals_direct_rating_loop(self):
        t=np.array([.2,.4,.8])
        q=s.population(t)
        mu=5*t
        ratings=np.array([[0,1,3],[2,3,2],[5,4,2]])
        out=s.empirical_metrics(q,mu,ratings)
        direct={k:[] for k in ('ce','brier','rps','nmse')}
        for i in range(3):
            for r in ratings[i]:
                direct['ce'].append(-np.log(q[i,r]))
                direct['brier'].append(sum((q[i,k]-int(r==k))**2 for k in range(6)))
                direct['rps'].append(np.mean([(sum(q[i,:k+1])-int(r<=k))**2 for k in range(5)]))
                direct['nmse'].append(((mu[i]-r)/5)**2)
        for key,values in direct.items():
            self.assertAlmostEqual(out[key],np.mean(values))
        # Sparse empirical CDF squared error omits outcome variance; it is not RPS.
        empirical=np.eye(6)[ratings].mean(axis=1)
        wrong=np.mean((np.cumsum(q-empirical,axis=1)[:,:-1])**2)
        self.assertGreater(out['rps'],wrong)

    def test_no_item_or_family_leakage(self):
        rows=s.items(CONFIG)
        self.assertEqual(len({r['id'] for r in rows}),len(rows))
        self.assertEqual(len({r['t'] for r in rows}),len(rows))
        splits={split:{r['family'] for r in rows if r['split']==split} for split in CONFIG['family_splits']}
        for a in splits:
            for b in splits:
                if a!=b:
                    self.assertFalse(splits[a]&splits[b])
        self.assertEqual({split:sum(r['split']==split for r in rows) for split in splits},
                         {'train':96,'validation':48,'test':48})

    def test_ties_undefined_correlations_and_permutation_invariance(self):
        t=np.linspace(.1,.9,17)
        q,mu=s.candidates(t,np.array([[1,2,3]]),CONFIG)
        ratings=np.rint(5*t).astype(int)[:,None]
        metrics=[s.empirical_metrics(qq,mm,ratings) for qq,mm in zip(q,mu)]
        for objective in ('pearson','spearman','kendall','nmse'):
            chosen=s.select(metrics,objective)
            self.assertEqual(chosen,list(range(5)))
            shuffled=list(reversed(metrics))
            self.assertEqual(sorted(len(metrics)-1-i for i in s.select(shuffled,objective)),chosen)
        self.assertEqual(s.select([{'pearson':None}],'pearson'),[])

    def test_sampler_and_deterministic_prefix(self):
        rng1=np.random.default_rng(np.random.SeedSequence([20261108,0,0]))
        rng2=np.random.default_rng(np.random.SeedSequence([20261108,0,0]))
        x=rng1.binomial(5,.3,size=50000)
        np.testing.assert_array_equal(x,rng2.binomial(5,.3,size=50000))
        np.testing.assert_allclose(np.bincount(x,minlength=6)/len(x),s.population(np.array([.3]))[0],atol=.008)

    def test_constant_fits_training_only(self):
        t=np.array([.2,.8])
        train=np.array([[0,1,2]])
        q,mu=s.candidates(t,train,CONFIG)
        expected=(np.bincount(train.ravel(),minlength=6)+1)/9
        np.testing.assert_allclose(q[-1],np.tile(expected,(2,1)))
        # Candidate API has no validation/test labels; all other candidates are independent of training outcomes.
        q2,mu2=s.candidates(t,np.array([[5,5,5]]),CONFIG)
        np.testing.assert_array_equal(q[:-1],q2[:-1])
        self.assertFalse(np.array_equal(q[-1],q2[-1]))


if __name__=='__main__':
    unittest.main()
