"""
Baseline v2: semua kandidat (SVC/NuSVC/LinearSVC x kernel + pembanding) dengan protokol yang
sama, hanya memakai DATA LATIH, dicatat ke MLflow (1 run per kandidat).

Jalankan dari root proyek:
    python -m experiments.run_baseline --quick      # uji cepat (bukan untuk kesimpulan)
    python -m experiments.run_baseline              # penuh
    python -m experiments.run_baseline --only svc_poly,nusvc_poly
Lalu:
    python -m experiments.select_baseline_winner
"""
import argparse
import json
import os
import tempfile
import warnings

import mlflow
import numpy as np
import pandas as pd

from src.baseline import build_candidates, evaluate_candidate
from src.data import CV_SEED, SPLIT_SEED, file_sha256, load_train
from src.metrics import PRIMARY, SCORING

warnings.filterwarnings("ignore")
EXPERIMENT = "CKD_Baseline_v2"


def already_done(name, mode):
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT], output_format="pandas",
        filter_string=f"tags.mode = '{mode}' and tags.mlflow.runName = '{name}' "
                      f"and attributes.status = 'FINISHED'")
    return len(runs) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/CKD.csv")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="", help="daftar nama kandidat dipisah koma")
    ap.add_argument("--force", action="store_true", help="jalankan ulang walau sudah ada hasilnya")
    args = ap.parse_args()
    mode = "quick" if args.quick else "full"

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT)

    X, y = load_train(args.data)          # hanya data latih; test set tidak dibuka
    sha = file_sha256(args.data)[:12]
    print(f"Mode={mode} | data latih={X.shape} | non-CKD(0)={np.mean(y == 0):.3f} CKD(1)={np.mean(y == 1):.3f}")
    print(f"Metrik pemilihan (PRIMARY) = {PRIMARY}\n")

    cands = build_candidates(y, quick=args.quick)
    if args.only:
        cands = {k: v for k, v in cands.items() if k in args.only.split(",")}

    for name, spec in cands.items():
        if not args.force and already_done(name, mode):
            print(f"[skip] {name} (sudah ada di MLflow, pakai --force untuk mengulang)")
            continue
        print(f"[run ] {name} ...", flush=True)
        fold_scores, best_params = evaluate_candidate(spec, X, y, quick=args.quick)
        n_failed = int(np.isnan(fold_scores[PRIMARY]).sum())

        with mlflow.start_run(run_name=name):
            mlflow.set_tags({"stage": "baseline", "mode": mode, "protocol": "v2",
                             "family": spec["family"]})
            mlflow.log_params({
                "svm_type": spec["svm_type"], "kernel": spec["kernel"],
                "preprocessing": "StandardScaler(kontinu)+passthrough(kategorikal)",
                "imbalance_handling": "none", "feature_selection": "none",
                "feature_engineering": False, "tuning_metric": PRIMARY,
                "inner_cv": "StratifiedKFold(3)",
                "outer_cv": "RepeatedStratifiedKFold(3x1)" if args.quick else "RepeatedStratifiedKFold(5x3)",
                "cv_seed": CV_SEED, "split_seed": SPLIT_SEED, "n_train": len(X),
                "minority_frac": round(float(np.mean(y == 0)), 4), "data_sha256": sha,
                "grid": json.dumps(spec["grid"], default=str)[:500],
            })
            mlflow.log_metric("n_failed_folds", n_failed)
            for m, s in fold_scores.items():
                mlflow.log_metric(f"{m}_mean", float(np.nanmean(s)))
                mlflow.log_metric(f"{m}_std", float(np.nanstd(s)))
                mlflow.log_metric(f"{m}_se", float(np.nanstd(s) / np.sqrt(np.sum(~np.isnan(s)))))
            with tempfile.TemporaryDirectory() as d:
                pd.DataFrame(fold_scores).to_csv(os.path.join(d, "fold_scores.csv"), index_label="fold")
                with open(os.path.join(d, "best_params_per_fold.json"), "w", encoding="utf-8") as f:
                    json.dump(best_params, f, default=str)
                mlflow.log_artifacts(d)
        print(f"        {PRIMARY} = {np.nanmean(fold_scores[PRIMARY]):.3f} ± {np.nanstd(fold_scores[PRIMARY]):.3f}"
              f"   (fold gagal: {n_failed})")

    print("\nSelesai. Berikutnya: python -m experiments.select_baseline_winner"
          + (" --mode quick" if args.quick else ""))


if __name__ == "__main__":
    main()
