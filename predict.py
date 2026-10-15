"""Predict OEM SOC from a CSV using a saved RF model."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent/'.packages'))
import argparse
import joblib
import pandas as pd

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('model',type=Path)
p.add_argument('input',type=Path,help='CSV with the four canonical feature columns')
p.add_argument('output',type=Path)
a=p.parse_args()
d=pd.read_csv(a.input)
cols=['voltage_V','current_A','battery_temperature_C','ambient_temperature_C']
if d[cols].isna().any().any():
    raise ValueError('Input features contain missing values')
d['predicted_soc_percent']=joblib.load(a.model).predict(d[cols].to_numpy(dtype='float32'))
d.to_csv(a.output,index=False)
