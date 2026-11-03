"""
CKD - Ablasi komponen : menguji strategi resampling, feature engineering, dan seleksi
fitur untuk kandidat terbaik dari baseline. Default menguji svc_poly dan nusvc_poly
(pemenang baseline), tapi bisa diarahkan ke kombinasi lain lewat argumen.

Jalankan dari root folder proyek:
    python experiments/run_ablation.py                                  (default: svc+nusvc, poly)
    python experiments/run_ablation.py --svm-type nusvc --kernel poly    (satu kandidat saja)
    python experiments/run_ablation.py --quick                           (uji cepat)
"""
import argparse
import os
import sys
import warnings

import mlflow
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pipeline import SCORING, PRIMARY_METRIC, build_pipeline, kernel_grid, nested_eval

warnings.filterwarnings("ignore")

EXPERIMENT_NAME = "CKD_SVM_Baseline"


def log_row(run_name, candidate_label, params, row):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.set_tags({"stage": "ablation_components", "candidate": candidate_label})
        for m in SCORING:
            mlflow.log_metric(f"{m}_mean", row[f"_{m}_mean"])
            mlflow.log_metric(f"{m}_std", row[f"_{m}_std"])
        print(f"  -> {run_name}: {PRIMARY_METRIC}={row[f'_{PRIMARY_METRIC}_mean']:.3f} "
              f"± {row[f'_{PRIMARY_METRIC}_std']:.3f}")


def ablate_one_candidate(X, y, svm_type, kernel, quick):
    candidate_label = f"{svm_type}_{kernel}"
    grid = kernel_grid(svm_type, kernel)
    print(f"\n--- Kandidat: {candidate_label} ---")

    configs = {
        "ref_smote_noFS_noFE": dict(resample="smote"),
        "resample_none": dict(resample="none"),
        "resample_class_weight": dict(resample="class_weight"),
        "feature_engineering": dict(resample="smote", fe=True),
        "fe_and_class_weight": dict(resample="class_weight", fe=True),
    }
    ks = [10, 20] if quick else [10, 15, 20, 30, 40]
    for k in ks:
        configs[f"FS_anova_k{k}"] = dict(fs=("anova", k))
        configs[f"FS_mi_k{k}"] = dict(fs=("mi", k))
    configs["FS_rfe_k20"] = dict(fs=("rfe", 20))

    for cfg_label, kwargs in configs.items():
        pipe = build_pipeline(X, svm_type=svm_type, kernel=kernel, **kwargs)
        row = nested_eval(f"{candidate_label}__{cfg_label}", pipe, grid, X, y, quick=quick)
        log_row(
            run_name=f"{candidate_label}__{cfg_label}",
            candidate_label=candidate_label,
            params={
                "svm_type": svm_type, "kernel": kernel,
                "resample": kwargs.get("resample", "smote"),
                "feature_engineering": kwargs.get("fe", False),
                "feature_selection": kwargs.get("fs", ("none", None))[0]
                    if kwargs.get("fs") else "none",
                "fs_k": kwargs.get("fs", (None, None))[1] if kwargs.get("fs") else None,
            },
            row=row,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/CKD.csv")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--svm-type", choices=["svc", "nusvc", "both"], default="both")
    parser.add_argument("--kernel", default="poly")
    args = parser.parse_args()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(args.data, sep=";").drop(
        columns=["PatientID", "DoctorInCharge"], errors="ignore")
    X, y = df.drop(columns="Diagnosis"), df["Diagnosis"]

    print(f"Data: {X.shape}")
    print(f"Mode: {'QUICK' if args.quick else 'PENUH'}")

    svm_types = ["svc", "nusvc"] if args.svm_type == "both" else [args.svm_type]
    for st in svm_types:
        ablate_one_candidate(X, y, st, args.kernel, args.quick)

    print("\nSelesai. Cek hasil di MLflow (tag stage=ablation_components).")


if __name__ == "__main__":
    main()
