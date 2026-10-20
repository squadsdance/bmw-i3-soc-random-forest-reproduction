"""Auditable reconstruction of Sulaiman & Mustaffa (2024), DOI 100177."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / ".packages"))
import argparse
import hashlib
import json
import platform
import time
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
URL = "https://www.kaggle.com/api/v1/datasets/download/atechnohazard/battery-and-heating-data-in-real-driving-cycles"
KEYS = [
    "Battery Voltage [",
    "Battery Current [",
    "Battery Temperature [",
    "Ambient Temperature [",
    "SoC [",
]
NAMES = [
    "voltage_V",
    "current_A",
    "battery_temperature_C",
    "ambient_temperature_C",
    "soc_percent",
]


# Optional: fetch the original archive if you want a fresh copy.
def download():
    base = ROOT / "data"
    base.mkdir(exist_ok=True)
    archive = base / "dataset.zip"
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    if not zipfile.is_zipfile(archive):
        raise ValueError("Download is not a ZIP; obtain the dataset from Kaggle manually.")
    with zipfile.ZipFile(archive) as z:
        target = (base / "raw").resolve()
        for name in z.namelist():
            if not (target / name).resolve().is_relative_to(target):
                raise ValueError("Unsafe archive member")
        z.extractall(target)
    return archive


# 1. LOAD AND CLEAN THE DATA
# Each CSV is one drive. Keep four readings and the recorded SOC (charge percentage).
def load_data():
    # Unpack the included data on the first run.
    if not (ROOT / "data/raw/TripA01.csv").exists():
        for archive in sorted((ROOT / "data").glob("trips_*.zip")):
            with zipfile.ZipFile(archive) as z:
                z.extractall(ROOT / "data/raw")
    # Sorting keeps the same trip order every time: A01-A32, then B01-B38.
    files = sorted((ROOT / "data/raw").rglob("Trip*.csv"))
    if len(files) != 70:
        raise ValueError(f"Expected 70 trips, found {len(files)}. Run with --download.")
    parts = []
    audit = []
    for trip_number, file_path in enumerate(files):
        # The original files use semicolons between values and Latin-1 text encoding.
        data = pd.read_csv(file_path, sep=";", encoding="latin1")
        # Find columns by their header names because some trips have fewer columns.
        columns = []
        for prefix in KEYS:
            for column in data.columns:
                if column.startswith(prefix):
                    columns.append(column)
                    break
            else:
                raise ValueError(f"Missing column: {prefix} in {file_path.name}")
        # Turn invalid values into NaN (missing), then discard incomplete rows.
        clean_data = (
            data[columns]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        clean_data.columns = NAMES
        valid = clean_data.notna().all(axis=1)
        clean_data = clean_data.loc[valid].copy()
        # Keep the trip, original line number and time so we can trace predictions.
        clean_data["trip"] = file_path.stem
        clean_data["source_row"] = np.flatnonzero(valid) + 2
        clean_data["time_s"] = pd.to_numeric(
            data.loc[valid, "Time [s]"], errors="coerce"
        ).values
        # Train on the first 60 trips; test on the last 10, which stay separate.
        if trip_number < 60:
            clean_data["split"] = "train"
        else:
            clean_data["split"] = "test"
        parts.append(clean_data)
        # Record row counts and a file fingerprint to identify the data used.
        audit.append(
            dict(
                trip=file_path.stem,
                split=clean_data["split"].iloc[0],
                raw_rows=len(data),
                retained_rows=len(clean_data),
                dropped_rows=len(data) - len(clean_data),
                sha256=hashlib.sha256(file_path.read_bytes()).hexdigest(),
            )
        )
    return (pd.concat(parts, ignore_index=True), pd.DataFrame(audit))


# 2. MEASURE PREDICTION ERRORS
# RMSE = square errors, average, square root. MAE = average absolute error.
# Both are in SOC percentage points. Lower is better.
def metrics(y, predictions_array):
    # Positive errors mean the model predicted too much charge.
    # std_error measures error spread; ddof=1 uses the sample standard deviation.
    errors = predictions_array - y
    return dict(
        rmse=float(np.sqrt(np.mean(errors**2))),
        mae=float(np.mean(abs(errors))),
        max_error=float(np.max(abs(errors))),
        std_error=float(np.std(errors, ddof=1)),
    )


# 3. ELM COMPARISON MODEL
# An extreme learning machine has random hidden weights and learned output weights.
# These are assumed settings because the paper does not give its exact ELM setup.
class ELM:
    """Assumed baseline: standardized inputs, sigmoid hidden layer, ridge output."""

    def __init__(self, seed, hidden=100, ridge=0.001):
        self.seed = seed
        self.hidden = hidden
        self.ridge = ridge

    def fit(self, x, y):
        # Learn means/scales from training inputs only, then reuse them on test inputs.
        self.scaler = StandardScaler().fit(x)
        # A seed repeats the same random weights. Hidden weights do not get trained.
        rng = np.random.default_rng(self.seed)
        self.w = rng.uniform(-1, 1, (x.shape[1], self.hidden))
        self.b = rng.uniform(-1, 1, self.hidden)
        # Accumulate H.T @ H and H.T @ y without storing a huge hidden-layer table.
        # @ means matrix multiplication; .T swaps rows and columns.
        feature_products = np.zeros((self.hidden + 1, self.hidden + 1))
        feature_targets = np.zeros(self.hidden + 1)
        # Work in chunks of 10,000 rows to keep memory use manageable.
        for start in range(0, len(x), 10000):
            hidden_outputs = self.design(x[start : start + 10000])
            feature_products += hidden_outputs.T @ hidden_outputs
            feature_targets += hidden_outputs.T @ y[start : start + 10000]
        # A small ridge penalty stabilises the solution. The intercept is not penalised.
        penalty = np.eye(self.hidden + 1) * self.ridge
        penalty[-1, -1] = 0
        # Solve for the output weights: (H.T H + penalty) beta = H.T y.
        self.beta = np.linalg.solve(feature_products + penalty, feature_targets)
        return self

    # Turn the four readings into sigmoid neuron outputs; add ones for an intercept.
    def design(self, x):
        scaled_inputs = self.scaler.transform(x)
        hidden_values = expit(scaled_inputs @ self.w + self.b)
        intercept_column = np.ones(len(x))
        return np.column_stack([hidden_values, intercept_column])

    # Multiply hidden outputs by learned output weights to predict each row.
    def predict(self, x):
        predictions = []
        # Work in chunks of 10,000 rows to keep memory use manageable.
        for start in range(0, len(x), 10000):
            chunk = x[start : start + 10000]
            chunk_predictions = self.design(chunk) @ self.beta
            predictions.append(chunk_predictions)
        return np.concatenate(predictions)


# 4. RANDOM FOREST SETTINGS
# A forest averages predictions from multiple decision trees.
# min_leaf: minimum samples per final leaf; max_features=0.5: two of four inputs
# at each split. Bootstrap samples rows with replacement. jobs controls workers.
# warm_start lets 25 trees grow to 50, 75 and 100 without rebuilding earlier trees.
def create_forest(settings, seed):
    return RandomForestRegressor(
        min_samples_leaf=settings.min_leaf,
        max_features=settings.max_features,
        bootstrap=True,
        random_state=seed,
        n_jobs=settings.jobs,
        warm_start=True,
    )


# 5. RUN THE EXPERIMENT
# Read options, prepare data, validate tree counts, train models and save results.
def main():
    # Defaults reproduce the full experiment; --trees 25 --runs 1 is a smaller run.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--trees", type=int, nargs="+", default=[25, 50, 75, 100])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--cv", choices=["row", "trip", "none"], default="row")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-leaf", type=int, default=5)
    parser.add_argument("--max-features", type=float, default=0.5)
    parser.add_argument("--elm-hidden", type=int, default=100)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--predict", nargs=3, metavar=("MODEL", "INPUT", "OUTPUT"))
    settings = parser.parse_args()
    # These optional modes replace the old separate prediction/report/check scripts.
    if settings.predict:
        predict_file(*settings.predict)
        return
    if settings.report_only:
        make_report(settings.output)
        return
    if settings.verify:
        verify_results(settings.output)
        return
    if settings.runs < 1 or any((t < 1 for t in settings.trees)):
        parser.error("runs and trees must be positive")
    if settings.download:
        download()
    output_folder = settings.output
    output_folder.mkdir(parents=True, exist_ok=True)
    data, audit = load_data()
    audit.to_csv(output_folder / "data_audit.csv", index=False)
    # Divide the cleaned data by trip before fitting any model.
    train = data[data.split == "train"]
    test = data[data.split == "test"]
    # x/y = training readings/answers; x_test/y_test = unseen readings/answers.
    # Each x row has four numbers. Each y row has one charge percentage.
    x = train[NAMES[:4]].to_numpy(dtype=np.float32)
    y = train.soc_percent.to_numpy()
    x_test = test[NAMES[:4]].to_numpy(dtype=np.float32)
    y_test = test.soc_percent.to_numpy()
    # Save settings and library versions so the experiment can be repeated.
    saved_settings = {}
    for name, value in vars(settings).items():
        if isinstance(value, Path):
            saved_settings[name] = str(value)
        else:
            saved_settings[name] = value
    manifest = dict(
        arguments=saved_settings,
        python=platform.python_version(),
        numpy=np.__version__,
        pandas=pd.__version__,
        sklearn=sklearn.__version__,
        train_rows=len(train),
        test_rows=len(test),
        train_trips=train.trip.unique().tolist(),
        test_trips=test.trip.unique().tolist(),
    )
    (output_folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Train {len(train):,}; test {len(test):,}", flush=True)
    # Keep scores and row-by-row predictions for every run.
    test_scores = []
    validation_scores = []
    validation_errors = {}
    predictions = test[["trip", "source_row", "time_s", "soc_percent"]].copy()
    # Repeat using different seeds; default seeds are 42, 43, 44, 45 and 46.
    for run in range(settings.runs):
        seed = settings.seed + run
        # Cross-validation uses only training data to choose the tree count.
        if settings.cv != "none":
            # Row CV mixes nearby samples; trip CV holds out entire drives instead.
            if settings.cv == "row":
                splitter = KFold(5, shuffle=True, random_state=seed)
                splits = splitter.split(x, y)
            else:
                splitter = GroupKFold(5)
                splits = splitter.split(x, y, groups=train.trip)
            # Train on four folds and validate on the fifth, rotating through all five.
            for fold, (training_indices, validation_indices) in enumerate(splits, 1):
                forest = create_forest(settings, seed)
                fold_inputs = x[training_indices]
                fold_answers = y[training_indices]
                validation_inputs = x[validation_indices]
                fit_seconds = 0
                # Grow the same forest in increasing sizes to avoid duplicate training.
                for trees in sorted(set(settings.trees)):
                    start = time.perf_counter()
                    # fit() learns from inputs and known answers.
                    forest.set_params(n_estimators=trees).fit(fold_inputs, fold_answers)
                    fit_seconds += time.perf_counter() - start
                    start = time.perf_counter()
                    # predict() only gets inputs. Compare its output with held-out answers.
                    score = metrics(y[validation_indices], forest.predict(validation_inputs))
                    validation_scores.append(
                        dict(
                            trees=trees,
                            run=run + 1,
                            fold=fold,
                            seconds=fit_seconds + time.perf_counter() - start,
                            **score,
                        )
                    )
                    if trees not in validation_errors:
                        validation_errors[trees] = []
                    validation_errors[trees].append(score["rmse"])
                    pd.DataFrame(validation_scores).to_csv(
                        output_folder / "cv_metrics.csv", index=False
                    )
                    print(
                        f"RF {trees}, run {run + 1}, fold {fold}: RMSE {score['rmse']:.4f}",
                        flush=True,
                    )
                del forest, fold_inputs, fold_answers, validation_inputs
        # Start a fresh forest, fit all training trips, then evaluate the test trips.
        forest = create_forest(settings, seed)
        fit_seconds = 0
        for trees in sorted(set(settings.trees)):
            start = time.perf_counter()
            forest.set_params(n_estimators=trees).fit(x, y)
            fit_seconds += time.perf_counter() - start
            start = time.perf_counter()
            # No test answers are given to predict(); they are used only for scoring.
            predictions_array = forest.predict(x_test)
            score = metrics(y_test, predictions_array)
            test_scores.append(
                dict(
                    model="RF",
                    trees=trees,
                    run=run + 1,
                    seed=seed,
                    seconds=fit_seconds + time.perf_counter() - start,
                    **score,
                )
            )
            predictions[f"RF_{trees}_run{run + 1}"] = predictions_array
            # Save the first seeded model for each size, not the luckiest test run.
            if run == 0:
                joblib.dump(forest, output_folder / f"rf_{trees}.joblib", compress=3)
            print(f"RF {trees}, run {run + 1}: test {score}", flush=True)
            pd.DataFrame(test_scores).to_csv(output_folder / "test_metrics.csv", index=False)
    # Repeat using different seeds; default seeds are 42, 43, 44, 45 and 46.
    for run in range(settings.runs):
        start = time.perf_counter()
        # Fit the ELM baseline on exactly the same training inputs and answers.
        elm_model = ELM(settings.seed + run, settings.elm_hidden).fit(x, y)
        predictions_array = elm_model.predict(x_test)
        test_scores.append(
            dict(
                model="ELM",
                trees=0,
                run=run + 1,
                seed=settings.seed + run,
                seconds=time.perf_counter() - start,
                **metrics(y_test, predictions_array),
            )
        )
        predictions[f"ELM_run{run + 1}"] = predictions_array
        if run == 0:
            joblib.dump(
                dict(
                    scaler=elm_model.scaler, w=elm_model.w, b=elm_model.b, beta=elm_model.beta
                ),
                output_folder / "elm_parameters.joblib",
            )
    pd.DataFrame(test_scores).to_csv(output_folder / "test_metrics.csv", index=False)
    predictions.to_csv(output_folder / "predictions.csv.gz", index=False)
    # Summarise minimum, average and variation across runs for each model.
    summary = (
        pd.DataFrame(test_scores)
        .groupby(["model", "trees"])[["rmse", "mae", "max_error", "std_error", "seconds"]]
        .agg(["min", "mean", "std"])
    )
    summary.to_csv(output_folder / "summary.csv")
    # Choose using validation error, never test error. Without CV use the first count.
    selected = settings.trees[0]
    selection_basis = "first requested tree count; CV skipped"
    average_validation_errors = {}
    lowest_error = float("inf")
    for trees, errors in validation_errors.items():
        average_error = float(np.mean(errors))
        average_validation_errors[trees] = average_error
        if average_error < lowest_error:
            lowest_error = average_error
            selected = trees
    if validation_errors:
        selection_basis = "mean validation RMSE"
    selection = {
        "trees": selected,
        "basis": selection_basis,
        "cv_rmse": average_validation_errors,
    }
    (output_folder / "selection.json").write_text(json.dumps(selection, indent=2))
    # Top graph: recorded and predicted charge. Bottom: predicted minus recorded.
    # The horizontal axis counts samples across trips, not elapsed seconds.
    fig, axs = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    index = np.arange(1, len(y_test) + 1)
    axs[0].plot(index, y_test, label="Observed OEM SOC", lw=1)
    for col in [f"RF_{selected}_run1", "ELM_run1"]:
        axs[0].plot(index, predictions[col], label=col, lw=0.7, alpha=0.85)
        axs[1].plot(index, predictions[col] - y_test, label=col, lw=0.6)
    axs[0].set_ylabel("SOC (%)")
    axs[1].set_ylabel("Predicted - observed (pp)")
    axs[1].set_xlabel("Retained test sample (trips concatenated; not elapsed seconds)")
    for ax in axs:
        ax.legend()
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_folder / "soc_comparison.png", dpi=160)
    plt.close(fig)
    print(summary.to_string(), flush=True)
    make_report(output_folder)
    verify_results(output_folder)


# 6. REPORT AND PAPER COMPARISON
# Use saved scores to make a table and chart; paper numbers are reference values.
def make_report(output_folder):
    scores = pd.read_csv(output_folder / "test_metrics.csv")
    paper = json.loads(Path(ROOT / "paper_results.json").read_text())
    manifest = json.loads((output_folder / "manifest.json").read_text())
    selection = json.loads((output_folder / "selection.json").read_text())
    records = []
    # Group repeated runs for each model. Minima for different metrics may differ in run.
    for (model, trees), runs in scores.groupby(["model", "trees"], sort=False):
        name = f"RF_{trees}" if model == "RF" else "ELM"
        for metric in paper["metric_order"]:
            metric_number = paper["metric_order"].index(metric)
            records.append(
                dict(
                    model=name,
                    metric=metric,
                    published_best=paper["best"][name][metric_number],
                    reproduced_min=runs[metric].min(),
                    published_average=paper["average"][name][metric_number],
                    reproduced_average=runs[metric].mean(),
                    across_run_std=runs[metric].std(),
                    average_delta=runs[metric].mean() - paper["average"][name][metric_number],
                )
            )
    comparison = pd.DataFrame(records)
    comparison.to_csv(output_folder / "published_comparison.csv", index=False)
    lines = [
        "# Reproduction results",
        "",
        "Measured results from the downloaded BMW i3 dataset. Published values are references, not substituted measurements.",
        "",
        f"Training samples: {manifest['train_rows']:,}; test samples: {manifest['test_rows']:,}. Runs per configuration: {manifest['arguments']['runs']}. CV: {manifest['arguments']['cv']}.",
        "",
        "## Held-out test results",
        "",
        "All errors are SOC percentage points. Best here means the minimum of that metric across runs, matching the table-style summary; minima can belong to different runs.",
        "",
        "| Model | Paper best RMSE | Measured min RMSE | Paper mean RMSE | Measured mean RMSE | Measured mean MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    # Group repeated runs for each model. Minima for different metrics may differ in run.
    for (model, trees), runs in scores.groupby(["model", "trees"], sort=False):
        name = f"RF_{trees}" if model == "RF" else "ELM"
        lines.append(
            f"| {name} | {paper['best'][name][0]:.4f} | {runs.rmse.min():.4f} | {paper['average'][name][0]:.4f} | {runs.rmse.mean():.4f} | {runs.mae.mean():.4f} |"
        )
    lines += [
        "",
        f"Selected tree count: {selection['trees']}, based on {selection['basis']}.",
        "",
        "## Validation and interpretation",
        "",
    ]
    if (output_folder / "cv_metrics.csv").exists():
        cv = pd.read_csv(output_folder / "cv_metrics.csv")
        for trees, runs in cv.groupby("trees"):
            lines.append(
                f"- {trees} trees: mean fold RMSE {runs.rmse.mean():.4f} across {len(runs)} fitted folds."
            )
    lines += [
        "",
        "Row-shuffled CV mixes neighboring samples from a trip and is optimistic relative to evaluation on unseen trips. Only training CV determines tree selection. The ten test trips remain separate.",
        "",
        "## Reproduction limits",
        "",
        "The cleaned test count matches the paper exactly. Training has one fewer row (945,026 versus 945,027). Raw time steps are approximately 0.1 seconds rather than the stated 1 Hz. RF leaf size, feature subsampling, depth, seeds, missing-data policy and folds are assumed. The ELM architecture is unspecified in the paper, so the implemented ELM is a transparent baseline rather than an exact reconstruction. MATLAB and scikit-learn implementations differ. Settings are recorded in manifest.json and main.py.",
        "",
        "The paper has inconsistent best/average entries and mixed decimal separators. Published values in paper_results.json are transcribed interpretations, not claims that the experiment was replicated bit-for-bit.",
        "",
        "## Figures",
        "",
        "![SOC and residuals](soc_comparison.png)",
        "",
        "![Published and measured mean RMSE](rmse_comparison.png)",
        "",
    ]
    (output_folder / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    rmse_rows = comparison[comparison.metric == "rmse"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(rmse_rows))
    ax.bar(
        [position - 0.19 for position in x],
        rmse_rows.published_average,
        width=0.38,
        label="Published mean",
    )
    ax.bar(
        [position + 0.19 for position in x],
        rmse_rows.reproduced_average,
        width=0.38,
        label="Measured mean",
    )
    ax.set_xticks(list(x), rmse_rows.model)
    ax.set_ylabel("Test RMSE (SOC percentage points)")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_folder / "rmse_comparison.png", dpi=160)
    plt.close(fig)
    print("Report saved to", output_folder / "REPORT.md")


# 7. CHECK SAVED RESULTS
# Check experiment coverage and recalculate each error from saved predictions.
def verify_results(output_folder):
    manifest = json.loads((output_folder / "manifest.json").read_text())
    settings = manifest["arguments"]
    scores = pd.read_csv(output_folder / "test_metrics.csv")
    predictions_array = pd.read_csv(output_folder / "predictions.csv.gz")
    # Require one result for each requested tree count and run, without duplicates.
    expected = {
        (t, run_number)
        for t in settings["trees"]
        for run_number in range(1, settings["runs"] + 1)
    }
    rf = scores[scores.model == "RF"]
    elm = scores[scores.model == "ELM"]
    assert set(zip(rf.trees, rf.run)) == expected and len(rf) == len(expected)
    assert len(elm) == settings["runs"] and elm.run.nunique() == settings["runs"]
    assert len(predictions_array) == manifest["test_rows"] == 118974
    assert np.isfinite(predictions_array.select_dtypes("number").to_numpy()).all()
    # Recalculate every metric and allow only tiny floating-point rounding differences.
    for row in scores.itertuples():
        col = f"RF_{row.trees}_run{row.run}" if row.model == "RF" else f"ELM_run{row.run}"
        recomputed = metrics(
            predictions_array.soc_percent.to_numpy(), predictions_array[col].to_numpy()
        )
        for name, value in recomputed.items():
            np.testing.assert_allclose(value, getattr(row, name), rtol=1e-10, atol=1e-10)
    if settings["cv"] != "none":
        cv = pd.read_csv(output_folder / "cv_metrics.csv")
        expected_cv = {
            (t, run_number, fold_number)
            for t, run_number in expected
            for fold_number in range(1, 6)
        }
        assert set(zip(cv.trees, cv.run, cv.fold)) == expected_cv and len(cv) == len(
            expected_cv
        )
    # Check audit counts and fingerprint format. This does not re-hash source files.
    audit = pd.read_csv(output_folder / "data_audit.csv")
    assert len(audit) == 70 and audit.sha256.str.len().eq(64).all()
    assert audit.groupby("split").retained_rows.sum().to_dict() == {
        "train": 945026,
        "test": 118974,
    }
    result = dict(
        status="passed",
        rf_test_models=len(rf),
        elm_test_models=len(elm),
        validation_fits=0 if settings["cv"] == "none" else len(cv),
        test_rows=len(predictions_array),
        checks=["coverage", "finite predictions", "recomputed metrics", "source audit"],
    )
    (output_folder / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


# 8. USE A SAVED FOREST
# Read four input columns in their training order and write predictions to a new CSV.
def predict_file(model_path, input_path, output_path):
    data = pd.read_csv(input_path)
    columns = ["voltage_V", "current_A", "battery_temperature_C", "ambient_temperature_C"]
    if data[columns].isna().any().any():
        raise ValueError("Input features contain missing values")
    data["predicted_soc_percent"] = joblib.load(model_path).predict(
        data[columns].to_numpy(dtype="float32")
    )
    data.to_csv(output_path, index=False)


# Start only when run directly. Importing this file does not start training.
if __name__ == "__main__":
    main()
