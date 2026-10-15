import unittest
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from reproduce import metrics,load_data,ELM

class ReproductionChecks(unittest.TestCase):
    def test_metric_units_and_residual_std(self):
        actual=metrics(np.array([0.,1.,2.]),np.array([1.,1.,0.]))
        self.assertAlmostEqual(actual['rmse'],np.sqrt(5/3))
        self.assertAlmostEqual(actual['mae'],1.)
        self.assertAlmostEqual(actual['max_error'],2.)
        self.assertAlmostEqual(actual['std_error'],np.sqrt(7/3))

    def test_incremental_forests_equal_separate_fits(self):
        rng=np.random.default_rng(12); x=rng.normal(size=(1000,4)); y=x[:,0]**2+x[:,1]
        params=dict(random_state=42,min_samples_leaf=5,max_features=.5,n_jobs=2)
        nested=RandomForestRegressor(n_estimators=25,warm_start=True,**params).fit(x,y)
        for trees in [25,50]:
            if trees>25: nested.set_params(n_estimators=trees).fit(x,y)
            separate=RandomForestRegressor(n_estimators=trees,**params).fit(x,y)
            np.testing.assert_allclose(nested.predict(x),separate.predict(x),rtol=1e-12,atol=1e-12)

    @unittest.skipUnless(Path('data/raw/TripA01.csv').exists(),'Public dataset not downloaded')
    def test_source_split_and_oem_target(self):
        d,a=load_data()
        self.assertEqual(d.groupby('split').size().to_dict(),{'train':945026,'test':118974})
        tr=d[d.split=='train']; te=d[d.split=='test']
        self.assertFalse(set(tr.trip)&set(te.trip))
        self.assertEqual(te.trip.unique().tolist(),[f'TripB{i}' for i in range(29,39)])
        self.assertAlmostEqual(d.iloc[0].soc_percent,86.9)  # displayed value is 100
        self.assertTrue(np.isfinite(d.iloc[:,:5].to_numpy()).all())
        self.assertEqual(int(a.raw_rows.sum()),1094793)

    def test_elm_handles_constant_feature(self):
        x=np.column_stack([np.linspace(-1,1,100),np.ones((100,3))])
        y=50+10*x[:,0]
        m=ELM(42,hidden=12).fit(x,y)
        self.assertTrue(np.isfinite(m.predict(x)).all())
        self.assertLess(metrics(y,m.predict(x))['rmse'],1.)

if __name__=='__main__': unittest.main()
