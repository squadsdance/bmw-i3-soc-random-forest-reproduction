"""Verify complete experiment coverage and recompute metrics from saved predictions."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent/'.packages'))
import argparse,json
import numpy as np
import pandas as pd
from reproduce import metrics

p=argparse.ArgumentParser(); p.add_argument('directory',type=Path,nargs='?',default=Path('results'))
a=p.parse_args(); out=a.directory
manifest=json.loads((out/'manifest.json').read_text()); args=manifest['arguments']
r=pd.read_csv(out/'test_metrics.csv'); pred=pd.read_csv(out/'predictions.csv.gz')
expected={(t,i) for t in args['trees'] for i in range(1,args['runs']+1)}
rf=r[r.model=='RF']; elm=r[r.model=='ELM']
assert set(zip(rf.trees,rf.run))==expected and len(rf)==len(expected)
assert len(elm)==args['runs'] and elm.run.nunique()==args['runs']
assert len(pred)==manifest['test_rows']==118974
assert np.isfinite(pred.select_dtypes('number').to_numpy()).all()
for row in r.itertuples():
    col=f'RF_{row.trees}_run{row.run}' if row.model=='RF' else f'ELM_run{row.run}'
    recomputed=metrics(pred.soc_percent.to_numpy(),pred[col].to_numpy())
    for name,value in recomputed.items():
        np.testing.assert_allclose(value,getattr(row,name),rtol=1e-10,atol=1e-10)
if args['cv']!='none':
    cv=pd.read_csv(out/'cv_metrics.csv')
    expected_cv={(t,i,f) for t,i in expected for f in range(1,6)}
    assert set(zip(cv.trees,cv.run,cv.fold))==expected_cv and len(cv)==len(expected_cv)
audit=pd.read_csv(out/'data_audit.csv')
assert len(audit)==70 and audit.sha256.str.len().eq(64).all()
assert audit.groupby('split').retained_rows.sum().to_dict()=={'train':945026,'test':118974}
result=dict(status='passed',rf_test_models=len(rf),elm_test_models=len(elm),
            validation_fits=0 if args['cv']=='none' else len(cv),test_rows=len(pred),
            checks=['coverage','finite predictions','recomputed metrics','source audit'])
(out/'verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
