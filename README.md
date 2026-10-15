# BMW i3 SOC estimation reproduction

Reconstruction of **State of charge estimation for electric vehicles using random forest**, Mohd Herwan Sulaiman and Zuriani Mustaffa, Green Energy and Intelligent Transportation 3 (2024), 100177. https://doi.org/10.1016/j.geits.2024.100177

This project uses the actual public driving dataset. It implements the reported RF experiment and an explicitly assumed ELM baseline. Exact numerical equivalence cannot be guaranteed because the paper does not release the executable experiment or several essential settings. Measured outputs are in result directories; `paper_results.json` contains separately labeled published reference numbers.

## Dataset and provenance

- Original: Matthias Steinstraeter, Johannes Buberger and Dimitar Trifonov, **Battery and Heating Data in Real Driving Cycles**, IEEE DataPort (2020), https://doi.org/10.21227/6jr9-5235
- Original landing page: https://ieee-dataport.org/open-access/battery-and-heating-data-real-driving-cycles
- Downloaded mirror: https://www.kaggle.com/datasets/atechnohazard/battery-and-heating-data-in-real-driving-cycles
- Public download endpoint is stored in `reproduce.py`. The IEEE page could not be read through the web tool; the public Kaggle ZIP downloaded successfully.
- Downloaded archive: `data/dataset.zip`; extracted originals: `data/raw/`. Preserve upstream terms and attribution when redistributing.
- 32 summer trips (`TripA01`–`TripA32`), 38 winter trips (`TripB01`–`TripB38`). CSVs are semicolon-separated and Latin-1 encoded. Column selection uses header prefixes, since some trips have fewer sensor columns and cannot safely use column positions.
- Each run writes per-file SHA-256 hashes and raw/retained/dropped row counts to `data_audit.csv`.

## Reconstructing the paper

Inputs: battery voltage, battery current, battery temperature, ambient temperature. Target: `SoC [%]`, the OEM SOC estimate, **not** `displayed SoC [%]`. This is reproduction of an OEM estimate, not independent electrochemical ground truth.

Order trips lexically, A01–A32 then B01–B38. First 60 trips train (through B28), last 10 trips test (B29–B38). Remove rows missing any of the four features or the target; no interpolation, no resampling, no use of test data for preprocessing.

| Split | Raw rows | Retained here | Paper |
|---|---:|---:|---:|
| Training | 971,402 | 945,026 | 945,027 |
| Test | 123,391 | 118,974 | 118,974 |

The test count matches exactly; training differs by one. Do not insert or duplicate a row to force agreement. The raw time column advances in approximately 0.1 s steps, while the paper describes 1 Hz. Preserve samples because their counts match the paper. Concatenated sample index is not elapsed time, especially after missing rows are removed.

RF: 25, 50, 75, 100 trees; bootstrap sampling; five seeds (42–46); five-fold CV within training data. Full-data models evaluate the held-out ten trips. Tree count selection uses mean CV RMSE, never held-out test RMSE. The fixed 25-tree result remains available even if CV selects another count.

### Explicit assumptions and limitations

The paper used MATLAB, but gives no exact function, seed, tree depth, leaf size, predictor subsampling, cleaning rules, fold assignments, or ELM architecture/activation/regularization. This Python reconstruction uses scikit-learn, unlimited depth, minimum leaf size 5 and two of four features per split (`--max-features 0.5`). These approximate MATLAB TreeBagger regression defaults; they are assumptions, not recovered author settings. See https://www.mathworks.com/help/stats/treebagger.html and https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html. Splitting and random-number implementations differ across MATLAB and Python.

Default CV shuffles individual rows, a reasonable interpretation of the paper's unspecified folds. Neighboring samples from the same trip can enter both training and validation, making CV optimistic. `--cv trip` performs a separate, stronger generalization check with entire trips held out; do not describe those scores as the paper's protocol.

ELM is a single hidden sigmoid layer with 100 random neurons, uniform [-1,1] weights/biases, train-only feature standardization, and ridge 0.001 on output weights with an unpenalized intercept. It is trained in chunks using accumulated normal equations. These settings are not reported in the paper. Its comparisons cannot establish an exact reproduction of the authors' ELM. No online updates or test-time learning occur in either model.

Metrics: RMSE, MAE, maximum absolute residual, and sample standard deviation of signed residuals (`ddof=1`); units are SOC percentage points. STD convention is unspecified in the paper. `seconds` measures final fit plus prediction, excludes CV/model serialization, and is not directly comparable to the paper's ambiguously scoped timings.

The printed Table 2 format `5.902,8` is interpreted as 5.9028. The table has internal inconsistencies: ELM average MAE is lower than its reported best MAE; some average maxima are below the reported best maxima; text also disagrees with table/figure values. Do not force reconstructed metrics to those entries. Summary minima are per metric, not necessarily from one common best run.

## Run

With Python 3.12+ in this folder:

```powershell
python -m pip install -r requirements.txt
python reproduce.py --download
python make_report.py results
```

Data is already downloaded here. The complete default experiment performs all four tree counts, five runs, five CV folds plus final fitting per run, and five ELM fits. This can take substantial time. A single configuration check:

```powershell
python reproduce.py --trees 25 --runs 1 --cv row --output results_initial
```

Separate trip-based validation:

```powershell
python reproduce.py --trees 25 --runs 1 --cv trip --output results_trip_cv
```

Use `--jobs` to set parallel tree-building workers. `--cv none` skips validation and labels selection accordingly. `--min-leaf`, `--max-features`, and `--elm-hidden` expose uncertain settings. The locally installed libraries live in `.packages`; scripts automatically use them. `requirements-lock.txt` records versions used for the actual runs.

On this Windows computer, where the standard `python` launcher is unavailable, `./run.ps1` uses the bundled Python. Forests grow incrementally through the requested tree counts, preserving the exact seeded tree sequence while avoiding duplicate fitting. Reported fit times accumulate tree growth and add only that configuration's prediction time.

Run checks with `python -m unittest test_reproduction.py` after installing dependencies. When using the local `.packages` installation, use `python -c "import reproduce,unittest; unittest.main(module='test_reproduction')"`. Checks verify metric units, incremental forest equivalence, trip separation, OEM target choice, missing-data removal, and ELM numerical behavior.

## Outputs

- `manifest.json`: versions, arguments, split counts, exact trip membership.
- `data_audit.csv`: source checksums and cleaning counts.
- `cv_metrics.csv`: each fold's metrics, saved as the run progresses.
- `test_metrics.csv`: final models' actual held-out results.
- `summary.csv`: per-configuration minima, means, and across-run standard deviations.
- `selection.json`: training CV based model selection.
- `predictions.csv.gz`: every test prediction with source trip, CSV row, timestamp, observed SOC.
- `rf_*.joblib`: first seeded RF for each tree count; load only trusted serialized models.
- `elm_parameters.joblib`: first ELM's scaler and learned parameters.
- `soc_comparison.png`: observed SOC, RF/ELM predictions, signed errors.

Predict from a new comma-separated CSV containing the four canonical columns listed in `predict.py`:

```powershell
python predict.py results/rf_25.joblib new_measurements.csv predicted.csv
```
