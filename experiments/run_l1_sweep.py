"""
CKD - Sweep C (alpha=1/C) untuk seleksi fitur L1, dengan aturan 1-SE.
Dipakai HANYA jika salah satu konfigurasi "FS_l1_*" di tahap ablasi komponen terlihat
menjanjikan. Kalau seleksi fitur (ANOVA/MI/RFE) semuanya kalah dari baseline tanpa FS,
langkah ini boleh dilewati.

Jalankan dari root folder proyek:
    python experiments/run_l1_sweep.py --svm-type svc --kernel poly
    python experiments/run_l1_sweep.py --svm-type svc --kernel poly --quick
"""
import argparse
import os
import sys
import warnings

import mlflow
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pipeline import l1_sweep

warnings.filterwarnings("ignore")

EXPERIMENT_NAME = "CKD_SVM_Baseline"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/CKD.csv")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--svm-type", choices=["svc", "nusvc"], required=True)
    parser.add_argument("--kernel", default="poly")
    parser.add_argument("--fe", action="store_true", help="Sertakan feature engineering")
    parser.add_argument("--resample", choices=["smote", "none", "class_weight"],
                        default="smote")
    args = parser.parse_args()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(args.data, sep=";").drop(
        columns=["PatientID", "DoctorInCharge"], errors="ignore")
    X, y = df.drop(columns="Diagnosis"), df["Diagnosis"]

    Cs = [0.01, 0.1] if args.quick else [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    candidate_label = f"{args.svm_type}_{args.kernel}"

    print(f"Sweep L1 untuk {candidate_label} (fe={args.fe}, resample={args.resample})")
    rows = l1_sweep(X, y, args.svm_type, args.kernel, Cs, fe=args.fe,
                    resample=args.resample, quick=args.quick)

    valid = [r for r in rows if not np.isnan(r["f1_mean"])]
    if not valid:
        print("Semua nilai C gagal (kemungkinan C terlalu kecil, semua fitur ternolkan).")
        return
    best = max(valid, key=lambda r: r["f1_mean"])
    ok = [r for r in valid if r["f1_mean"] >= best["f1_mean"] - best["f1_se"]]
    one_se_pick = min(ok, key=lambda r: r["C"])  # C terkecil = model paling sederhana

    for r in rows:
        with mlflow.start_run(run_name=f"{candidate_label}__L1_C{r['C']}"):
            mlflow.log_params({
                "svm_type": args.svm_type, "kernel": args.kernel,
                "feature_selection": "l1", "C": r["C"], "alpha": r["alpha"],
                "feature_engineering": args.fe, "resample": args.resample,
            })
            mlflow.set_tags({"stage": "l1_sweep", "candidate": candidate_label,
                             "one_se_pick": str(r is one_se_pick)})
            mlflow.log_metric("f1_macro_mean", r["f1_mean"] if not np.isnan(r["f1_mean"]) else -1)
            mlflow.log_metric("f1_macro_se", r["f1_se"] if not np.isnan(r["f1_se"]) else -1)
            mlflow.log_metric("n_features", r["n_features"])
            tag = " <- one_se_pick" if r is one_se_pick else ""
            print(f"  C={r['C']:<6} n_feat={r['n_features']:.1f}  "
                  f"f1={r['f1_mean']:.3f}{tag}")

    print(f"\nRekomendasi (aturan 1-SE): C={one_se_pick['C']}, "
         f"n_features≈{one_se_pick['n_features']:.0f}, f1={one_se_pick['f1_mean']:.3f}")


if __name__ == "__main__":
    main()
