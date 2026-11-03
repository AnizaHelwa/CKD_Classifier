"""
CKD - Baseline : membandingkan varian SVM (SVC, NuSVC, LinearSVC) x semua kernel yang
relevan, ditambah model pembanding (Logistic Regression, Random Forest, Dummy).
Semua konfigurasi di-tuning via nested CV dan di-log ke MLflow sebagai satu run per
konfigurasi (bukan per fold).

Jalankan dari root folder proyek:
    python experiments/run_baseline.py                  (penuh)
    python experiments/run_baseline.py --quick            (uji cepat)
"""
import argparse
import os
import sys
import tempfile
import warnings

import mlflow
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pipeline import (CAT, SCORING, PRIMARY_METRIC, SEED, build_pipeline,
                          kernel_grid, nested_eval, cv_objects, _cont_cols)

warnings.filterwarnings("ignore")

EXPERIMENT_NAME = "CKD_SVM_Baseline"

# Kombinasi (svm_type, kernel) yang diuji di baseline.
SVM_CONFIGS = [
    ("linearsvc", "linear"),
    ("svc", "linear"), ("svc", "poly"), ("svc", "rbf"), ("svc", "sigmoid"),
    ("nusvc", "linear"), ("nusvc", "poly"), ("nusvc", "rbf"), ("nusvc", "sigmoid"),
]


def log_run(run_name, params, row, extra_tags=None):
    """Log satu konfigurasi (ringkasan CV) sebagai satu MLflow run."""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        if extra_tags:
            mlflow.set_tags(extra_tags)
        for m in SCORING:
            mlflow.log_metric(f"{m}_mean", row[f"_{m}_mean"])
            mlflow.log_metric(f"{m}_std", row[f"_{m}_std"])
        # simpan daftar best_params per fold sebagai artifact teks, untuk audit
        if "_best_params_per_fold" in row:
            # pakai tempfile, bukan path hardcoded "/tmp/...", supaya jalan di Windows/Mac/Linux
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as f:
                f.write(str(row["_best_params_per_fold"]))
                tmp_path = f.name
            mlflow.log_artifact(tmp_path, artifact_path="best_params")
            os.remove(tmp_path)
        print(f"  -> {run_name}: {PRIMARY_METRIC}={row[f'_{PRIMARY_METRIC}_mean']:.3f} "
              f"± {row[f'_{PRIMARY_METRIC}_std']:.3f}")


def run_svm_baseline(X, y, quick):
    for svm_type, kernel in SVM_CONFIGS:
        label = f"{svm_type}_{kernel}"
        pipe = build_pipeline(X, svm_type=svm_type, kernel=kernel)
        grid = kernel_grid(svm_type, kernel)
        row = nested_eval(label, pipe, grid, X, y, quick=quick)
        log_run(
            run_name=label,
            params={"model_family": "SVM", "svm_type": svm_type, "kernel": kernel,
                    "resample": "smote", "feature_engineering": False,
                    "feature_selection": "none"},
            row=row,
            extra_tags={"stage": "baseline"},
        )


def _simple_pipeline(estimator):
    """Pipeline sederhana (scaling saja, tanpa SMOTE/FS/FE) untuk model pembanding."""
    cat = CAT
    return SkPipeline([
        ("prep", ColumnTransformer(
            [("num", StandardScaler(), _cont_cols), ("cat", "passthrough", cat)])),
        ("clf", estimator),
    ])


def run_comparison_models(X, y, quick):
    _, outer = cv_objects(quick)

    comparisons = {
        "dummy_most_frequent": (DummyClassifier(strategy="most_frequent"), {}),
        "logistic_regression": (
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED),
            {"clf__C": np.logspace(-2, 2, 3 if quick else 5)},
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
            {"clf__n_estimators": [200] if quick else [100, 200, 400],
             "clf__max_depth": [None, 10, 20]},
        ),
    }

    for label, (estimator, grid) in comparisons.items():
        pipe = _simple_pipeline(estimator)
        if grid:
            inner, _ = cv_objects(quick)
            model = GridSearchCV(pipe, grid, scoring=PRIMARY_METRIC, cv=inner, n_jobs=-1)
        else:
            model = pipe
        res = cross_validate(model, X, y, cv=outer, scoring=SCORING, n_jobs=1)
        row = {"config": label}
        for m in SCORING:
            s = res[f"test_{m}"]
            row[f"_{m}_mean"] = s.mean()
            row[f"_{m}_std"] = s.std()
        log_run(
            run_name=label,
            params={"model_family": "comparison", "svm_type": "-", "kernel": "-",
                    "resample": "-", "feature_engineering": False,
                    "feature_selection": "none"},
            row=row,
            extra_tags={"stage": "baseline_comparison"},
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/CKD.csv")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    # MLflow >= 3.x: backend file (./mlruns) sudah di-deprecate, pakai SQLite lokal.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(args.data, sep=";").drop(
        columns=["PatientID", "DoctorInCharge"], errors="ignore")
    X, y = df.drop(columns="Diagnosis"), df["Diagnosis"]

    print(f"Data: {X.shape}, kelas: {y.value_counts(normalize=True).to_dict()}")
    print(f"Mode: {'QUICK' if args.quick else 'PENUH'}\n")

    print("== Baseline model pembanding (Dummy, Logistic Regression, Random Forest) ==")
    run_comparison_models(X, y, args.quick)

    print("\n== Baseline SVM (SVC/NuSVC/LinearSVC x semua kernel) ==")
    run_svm_baseline(X, y, args.quick)

    print("\nSelesai. Jalankan 'mlflow ui' lalu buka http://localhost:5000 untuk melihat hasil.")


if __name__ == "__main__":
    main()