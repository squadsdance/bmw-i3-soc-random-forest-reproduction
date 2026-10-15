"""Summarize completed reproduction outputs without substituting paper values."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent/'.packages'))
import argparse,json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('directory',type=Path,nargs='?',default=Path('results'))
a=p.parse_args(); out=a.directory
r=pd.read_csv(out/'test_metrics.csv')
ref=json.loads(Path('paper_results.json').read_text())
manifest=json.loads((out/'manifest.json').read_text())
selection=json.loads((out/'selection.json').read_text())
records=[]
for (model,trees),g in r.groupby(['model','trees'],sort=False):
    name=f'RF_{trees}' if model=='RF' else 'ELM'
    for metric in ref['metric_order']:
        j=ref['metric_order'].index(metric)
        records.append(dict(model=name,metric=metric,published_best=ref['best'][name][j],
                            reproduced_min=g[metric].min(),published_average=ref['average'][name][j],
                            reproduced_average=g[metric].mean(),across_run_std=g[metric].std(),
                            average_delta=g[metric].mean()-ref['average'][name][j]))
c=pd.DataFrame(records); c.to_csv(out/'published_comparison.csv',index=False)
lines=['# Reproduction results','',
       'Measured results from the downloaded BMW i3 dataset. Published values are references, not substituted measurements.','',
       f'Training samples: {manifest["train_rows"]:,}; test samples: {manifest["test_rows"]:,}. '
       f'Runs per configuration: {manifest["arguments"]["runs"]}. CV: {manifest["arguments"]["cv"]}.','',
       '## Held-out test results','',
       'All errors are SOC percentage points. Best here means the minimum of that metric across runs, matching the table-style summary; minima can belong to different runs.','',
       '| Model | Paper best RMSE | Measured min RMSE | Paper mean RMSE | Measured mean RMSE | Measured mean MAE |',
       '|---|---:|---:|---:|---:|---:|']
for (model,trees),g in r.groupby(['model','trees'],sort=False):
    name=f'RF_{trees}' if model=='RF' else 'ELM'
    lines.append(f'| {name} | {ref["best"][name][0]:.4f} | {g.rmse.min():.4f} | {ref["average"][name][0]:.4f} | {g.rmse.mean():.4f} | {g.mae.mean():.4f} |')
lines += ['',f'Selected tree count: {selection["trees"]}, based on {selection["basis"]}.',
          '', '## Validation and interpretation', '']
if (out/'cv_metrics.csv').exists():
    cv=pd.read_csv(out/'cv_metrics.csv')
    for trees,g in cv.groupby('trees'):
        lines.append(f'- {trees} trees: mean fold RMSE {g.rmse.mean():.4f} across {len(g)} fitted folds.')
lines += ['',
          'Row-shuffled CV mixes neighboring samples from a trip and is optimistic relative to evaluation on unseen trips. Only training CV determines tree selection. The ten test trips remain separate.',
          '', '## Reproduction limits','',
          'The cleaned test count matches the paper exactly. Training has one fewer row (945,026 versus 945,027). '
          'Raw time steps are approximately 0.1 seconds rather than the stated 1 Hz. '
          'RF leaf size, feature subsampling, depth, seeds, missing-data policy and folds are assumed. '
          'The ELM architecture is unspecified in the paper, so the implemented ELM is a transparent baseline rather than an exact reconstruction. '
          'MATLAB and scikit-learn implementations differ. See README.md for all choices and sources.',
          '', 'The paper has inconsistent best/average entries and mixed decimal separators. '
          'Published values in paper_results.json are transcribed interpretations, not claims that the experiment was replicated bit-for-bit.',
          '', '## Figures','', '![SOC and residuals](soc_comparison.png)','',
          '![Published and measured mean RMSE](rmse_comparison.png)','']
(out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
v=c[c.metric=='rmse']; fig,ax=plt.subplots(figsize=(9,4.5))
x=range(len(v)); ax.bar([i-.19 for i in x],v.published_average,width=.38,label='Published mean')
ax.bar([i+.19 for i in x],v.reproduced_average,width=.38,label='Measured mean')
ax.set_xticks(list(x),v.model); ax.set_ylabel('Test RMSE (SOC percentage points)')
ax.legend(); ax.grid(axis='y',alpha=.2); fig.tight_layout()
fig.savefig(out/'rmse_comparison.png',dpi=160)
print((out/'REPORT.md').read_text())
