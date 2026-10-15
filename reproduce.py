"""Auditable reconstruction of Sulaiman & Mustaffa (2024), DOI 100177."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / '.packages'))
import argparse, hashlib, json, platform, time, urllib.request, zipfile
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
import joblib

ROOT = Path(__file__).parent
URL = 'https://www.kaggle.com/api/v1/datasets/download/atechnohazard/battery-and-heating-data-in-real-driving-cycles'
KEYS = ['Battery Voltage [', 'Battery Current [', 'Battery Temperature [', 'Ambient Temperature [', 'SoC [']
NAMES = ['voltage_V', 'current_A', 'battery_temperature_C', 'ambient_temperature_C', 'soc_percent']

def download():
    base = ROOT / 'data'
    base.mkdir(exist_ok=True)
    archive = base / 'dataset.zip'
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    if not zipfile.is_zipfile(archive):
        raise ValueError('Download is not a ZIP; obtain the dataset from Kaggle manually.')
    with zipfile.ZipFile(archive) as z:
        target = (base / 'raw').resolve()
        for name in z.namelist():
            if not (target / name).resolve().is_relative_to(target):
                raise ValueError('Unsafe archive member')
        z.extractall(target)
    return archive

def load_data():
    if not (ROOT / 'data/raw/TripA01.csv').exists():
        for archive in sorted((ROOT / 'data').glob('trips_*.zip')):
            with zipfile.ZipFile(archive) as z:
                z.extractall(ROOT / 'data/raw')
    files = sorted((ROOT / 'data/raw').rglob('Trip*.csv'))
    if len(files) != 70:
        raise ValueError(f'Expected 70 trips, found {len(files)}. Run with --download.')
    parts, audit = [], []
    for i, f in enumerate(files):
        d = pd.read_csv(f, sep=';', encoding='latin1')
        cols = [next(c for c in d if c.startswith(k)) for k in KEYS]
        v = d[cols].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf],np.nan)
        v.columns = NAMES
        valid = v.notna().all(axis=1)
        v = v.loc[valid].copy()
        v['trip'] = f.stem
        v['source_row'] = np.flatnonzero(valid) + 2  # CSV includes header
        v['time_s'] = pd.to_numeric(d.loc[valid, 'Time [s]'], errors='coerce').values
        v['split'] = 'train' if i < 60 else 'test'
        parts.append(v)
        audit.append(dict(trip=f.stem, split=v['split'].iloc[0], raw_rows=len(d),
                          retained_rows=len(v), dropped_rows=len(d)-len(v),
                          sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    return pd.concat(parts, ignore_index=True), pd.DataFrame(audit)

def metrics(y, pred):
    e = pred - y
    return dict(rmse=float(np.sqrt(np.mean(e**2))), mae=float(np.mean(abs(e))),
                max_error=float(np.max(abs(e))), std_error=float(np.std(e,ddof=1)))

class ELM:
    """Assumed baseline: standardized inputs, sigmoid hidden layer, ridge output."""
    def __init__(self, seed, hidden=100, ridge=1e-3):
        self.seed, self.hidden, self.ridge = seed, hidden, ridge
    def fit(self, x, y):
        self.scaler = StandardScaler().fit(x)
        rng = np.random.default_rng(self.seed)
        self.w = rng.uniform(-1,1,(x.shape[1],self.hidden))
        self.b = rng.uniform(-1,1,self.hidden)
        gram = np.zeros((self.hidden+1,self.hidden+1))
        rhs = np.zeros(self.hidden+1)
        for start in range(0,len(x),10000):
            h = self.design(x[start:start+10000])
            gram += h.T @ h
            rhs += h.T @ y[start:start+10000]
        reg = np.eye(self.hidden+1)*self.ridge
        reg[-1,-1] = 0
        self.beta = np.linalg.solve(gram+reg,rhs)
        return self
    def design(self,x):
        return np.column_stack([expit(self.scaler.transform(x) @ self.w + self.b),np.ones(len(x))])
    def predict(self,x):
        return np.concatenate([self.design(x[i:i+10000]) @ self.beta for i in range(0,len(x),10000)])

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download',action='store_true')
    p.add_argument('--trees',type=int,nargs='+',default=[25,50,75,100])
    p.add_argument('--runs',type=int,default=5)
    p.add_argument('--cv',choices=['row','trip','none'],default='row')
    p.add_argument('--jobs',type=int,default=8)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--min-leaf',type=int,default=5)
    p.add_argument('--max-features',type=float,default=0.5)
    p.add_argument('--elm-hidden',type=int,default=100)
    p.add_argument('--output',type=Path,default=ROOT/'results')
    a=p.parse_args()
    if a.runs < 1 or any(t<1 for t in a.trees): p.error('runs and trees must be positive')
    if a.download: download()
    out=a.output; out.mkdir(parents=True,exist_ok=True)
    d,audit=load_data(); audit.to_csv(out/'data_audit.csv',index=False)
    train=d[d.split=='train']; test=d[d.split=='test']
    x=train[NAMES[:4]].to_numpy(dtype=np.float32); y=train.soc_percent.to_numpy()
    xt=test[NAMES[:4]].to_numpy(dtype=np.float32); yt=test.soc_percent.to_numpy()
    manifest=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                  python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                  sklearn=sklearn.__version__,train_rows=len(train),test_rows=len(test),
                  train_trips=train.trip.unique().tolist(),test_trips=test.trip.unique().tolist())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Train {len(train):,}; test {len(test):,}',flush=True)
    rows=[]; cvrows=[]; best_cv={}; predictions=test[['trip','source_row','time_s','soc_percent']].copy()
    # Identical seeded forests are nested: retain existing trees when growing
    # 25 -> 50 -> 75 -> 100 instead of rebuilding the same prefixes each time.
    for run in range(a.runs):
        seed=a.seed+run
        def model():
            return RandomForestRegressor(min_samples_leaf=a.min_leaf,
                max_features=a.max_features,bootstrap=True,random_state=seed,
                n_jobs=a.jobs,warm_start=True)
        if a.cv!='none':
            splitter=KFold(5,shuffle=True,random_state=seed) if a.cv=='row' else GroupKFold(5)
            splits=splitter.split(x,y) if a.cv=='row' else splitter.split(x,y,groups=train.trip)
            for fold,(tr,va) in enumerate(splits,1):
                m=model(); trainx,trainy=x[tr],y[tr]; validx=x[va]
                fit_seconds=0
                for trees in sorted(set(a.trees)):
                    start=time.perf_counter(); m.set_params(n_estimators=trees).fit(trainx,trainy)
                    fit_seconds+=time.perf_counter()-start
                    start=time.perf_counter(); score=metrics(y[va],m.predict(validx))
                    cvrows.append(dict(trees=trees,run=run+1,fold=fold,
                        seconds=fit_seconds+time.perf_counter()-start,**score))
                    best_cv.setdefault(trees,[]).append(score['rmse'])
                    pd.DataFrame(cvrows).to_csv(out/'cv_metrics.csv',index=False)
                    print(f'RF {trees}, run {run+1}, fold {fold}: RMSE {score["rmse"]:.4f}',flush=True)
                del m,trainx,trainy,validx
        m=model(); fit_seconds=0
        for trees in sorted(set(a.trees)):
            start=time.perf_counter(); m.set_params(n_estimators=trees).fit(x,y)
            fit_seconds+=time.perf_counter()-start
            start=time.perf_counter(); pred=m.predict(xt)
            score=metrics(yt,pred)
            rows.append(dict(model='RF',trees=trees,run=run+1,seed=seed,seconds=fit_seconds+time.perf_counter()-start,**score))
            predictions[f'RF_{trees}_run{run+1}']=pred
            if run==0: joblib.dump(m,out/f'rf_{trees}.joblib',compress=3)
            print(f'RF {trees}, run {run+1}: test {score}',flush=True)
            pd.DataFrame(rows).to_csv(out/'test_metrics.csv',index=False)
    for run in range(a.runs):
        start=time.perf_counter(); m=ELM(a.seed+run,a.elm_hidden).fit(x,y); pred=m.predict(xt)
        rows.append(dict(model='ELM',trees=0,run=run+1,seed=a.seed+run,seconds=time.perf_counter()-start,**metrics(yt,pred)))
        predictions[f'ELM_run{run+1}']=pred
        if run==0: joblib.dump(dict(scaler=m.scaler,w=m.w,b=m.b,beta=m.beta),out/'elm_parameters.joblib')
    pd.DataFrame(rows).to_csv(out/'test_metrics.csv',index=False)
    predictions.to_csv(out/'predictions.csv.gz',index=False)
    summary=pd.DataFrame(rows).groupby(['model','trees'])[['rmse','mae','max_error','std_error','seconds']].agg(['min','mean','std'])
    summary.to_csv(out/'summary.csv')
    selected=min(best_cv,key=lambda t:np.mean(best_cv[t])) if best_cv else a.trees[0]
    (out/'selection.json').write_text(json.dumps(dict(trees=selected,basis='mean validation RMSE' if best_cv else 'first requested tree count; CV skipped',cv_rmse={t:np.mean(v) for t,v in best_cv.items()}),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,1,figsize=(13,7),sharex=True)
    index=np.arange(1,len(yt)+1)
    axs[0].plot(index,yt,label='Observed OEM SOC',lw=1)
    for col in [f'RF_{selected}_run1','ELM_run1']:
        axs[0].plot(index,predictions[col],label=col,lw=.7,alpha=.85)
        axs[1].plot(index,predictions[col]-yt,label=col,lw=.6)
    axs[0].set_ylabel('SOC (%)'); axs[1].set_ylabel('Predicted - observed (pp)')
    axs[1].set_xlabel('Retained test sample (trips concatenated; not elapsed seconds)')
    for ax in axs: ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out/'soc_comparison.png',dpi=160); plt.close(fig)
    print(summary.to_string(),flush=True)

if __name__=='__main__': main()
