"""
Tahap 2a - Screening komponen untuk PEMENANG baseline (dibaca otomatis dari results/baseline_winner.json).
Tiap komponen (normalisasi, encoding, penanganan imbalance, feature engineering, seleksi fitur) diuji
SATU PER SATU terhadap konfigurasi baseline, dengan protokol & fold yang sama dengan baseline.

    python -m experiments.run_screening --quick     # uji cepat (bukan untuk kesimpulan)
    python -m experiments.run_screening             # penuh
    python -m experiments.run_screening --only imbalance,feature_selection
"""
import argparse
import json
import warnings

import mlflow
import numpy as np
from joblib import Memory

from src.baseline import build_candidates
from src.data import load_train
from src.features import created_features
from src.stage2 import BASE_CFG, evaluate_config, screening_configs
from src.tracking import already_done, log_run
from src.winner import load_winner

warnings.filterwarnings("ignore")
EXPERIMENT = "CKD_Stage2_v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/CKD.csv")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cache", action="store_true", help="aktifkan cache joblib (lebih cepat di Linux/Mac; di Windows bisa PermissionError)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="", help="kelompok dipisah koma: scaler,encoding,imbalance,feature_engineering,feature_selection")
    args = ap.parse_args()
    mode = "quick" if args.quick else "full"

    winner = load_winner()
    if winner["mode"] == "quick" and mode == "full":
        raise SystemExit("baseline_winner.json berasal dari mode quick. Jalankan baseline penuh dulu.")
    wname = winner["winner"]["name"]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT)
    X, y = load_train(args.data)
    spec = build_candidates(y, quick=args.quick)[wname]
    print(f"Pemenang baseline: {wname} | mode={mode} | data latih={X.shape}")
    print(f"Fitur hasil feature engineering yang benar-benar terbentuk: {created_features(X)}\n")

    memory = Memory(location=".cache/pipeline", verbose=0) if args.cache else None
    jobs = [("reference", "baseline", BASE_CFG)] + screening_configs(args.quick)
    if args.only:
        keep = set(args.only.split(","))
        jobs = [j for j in jobs if j[0] == "reference" or j[0] in keep]

    for group, level, cfg in jobs:
        name = f"{group}:{level}"
        if not args.force and already_done(EXPERIMENT, name, mode, "screening"):
            print(f"[skip] {name}")
            continue
        if spec["svm_type"] == "nusvc" and cfg.get("resample") == "class_weight":
            print(f"[skip] {name} (class_weight tidak berpengaruh pada NuSVC)")
            continue
        print(f"[run ] {name} ...", flush=True)
        fold_scores, best_params, extra = evaluate_config(spec, X, y, cfg, quick=args.quick, memory=memory)
        if memory is not None:
            memory.clear(warn=False)
        log_run(name,
                tags={"stage": "screening", "mode": mode, "group": group, "level": level,
                      "winner": wname},
                params={**BASE_CFG, **cfg, "cfg_json": json.dumps({**BASE_CFG, **cfg}),
                        "winner": wname},
                fold_scores=fold_scores, best_params=best_params, extra=extra)
        m = fold_scores["mcc"]
        print(f"        mcc = {np.nanmean(m):.3f} ± {np.nanstd(m):.3f}  | gagal: {int(np.isnan(m).sum())}"
              f" | fitur akhir: {extra['n_features_final']}")

        if group == "reference":
            base = winner["winner"]["metrics"]["mcc"]
            diff = abs(np.nanmean(m) - base)
            tag = "OK" if diff < 0.005 else "PERHATIAN: tidak cocok dengan baseline!"
            print(f"        cek konsistensi vs baseline ({base:.3f}): selisih {diff:.4f}  {tag}")

    print("\nSelesai. Berikutnya: python -m experiments.select_components" + (" --mode quick" if args.quick else ""))


if __name__ == "__main__":
    main()
