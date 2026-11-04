"""
Tahap 2c - Validasi konfigurasi GABUNGAN (hasil select_components) lawan baseline dan komponen tunggal terbaik.
Pilihan final = konfigurasi dengan MCC rata-rata tertinggi di antara: baseline, komponen tunggal terbaik, gabungan.
Perbandingan ke baseline memakai fold yang sama + corrected resampled t-test (Nadeau & Bengio, 2003).
Tambahan: ablasi leave-one-out pada konfigurasi gabungan (lepas satu komponen sekali) untuk melihat
kontribusi tiap komponen DI DALAM kombinasi. Ini analisis untuk pembahasan, TIDAK dipakai memilih final.

    python -m experiments.run_combined [--quick]
"""
import argparse
import json
import warnings

import mlflow
import numpy as np
from joblib import Memory

from src.baseline import build_candidates
from src.data import load_train
from src.metrics import PRIMARY
from src.stage2 import BASE_CFG, evaluate_config
from src.stats import corrected_resampled_ttest
from src.tracking import find_run, load_fold_scores, log_run
from src.winner import load_winner

warnings.filterwarnings("ignore")
EXPERIMENT = "CKD_Stage2_v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/CKD.csv")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cache", action="store_true", help="aktifkan cache joblib (lebih cepat di Linux/Mac; di Windows bisa PermissionError)")
    ap.add_argument("--no-loo", action="store_true", help="lewati ablasi leave-one-out")
    ap.add_argument("--selection", default="results/stage2_selection.json")
    ap.add_argument("--out", default="results/stage2_final_config.json")
    args = ap.parse_args()
    mode = "quick" if args.quick else "full"

    sel = json.load(open(args.selection, encoding="utf-8"))
    if sel["mode"] != mode:
        raise SystemExit(f"stage2_selection.json bermode '{sel['mode']}', tetapi run ini '{mode}'.")
    wname = load_winner()["winner"]["name"]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT)
    X, y = load_train(args.data)
    spec = build_candidates(y, quick=args.quick)[wname]
    memory = Memory(location=".cache/pipeline", verbose=0) if args.cache else None

    ref_run = find_run(EXPERIMENT, "reference:baseline", mode, "screening")
    ref_scores = load_fold_scores(ref_run["run_id"])
    ratio = 0.5 if args.quick else 0.25      # n_test/n_train per fold (3-fold: 1/2, 5-fold: 1/4)

    candidates = {"baseline": dict(BASE_CFG)}
    if sel.get("best_single"):
        candidates["best_single"] = {**BASE_CFG, **sel["best_single"]["cfg"]}
    if sel["combined_cfg"] != BASE_CFG and sel["combined_cfg"] != candidates.get("best_single"):
        candidates["combined"] = sel["combined_cfg"]

    results = {"baseline": {"mcc": float(ref_scores[PRIMARY].mean()), "cfg": candidates["baseline"]}}
    fold_store = {}
    for label in [k for k in candidates if k != "baseline"]:
        cfg = candidates[label]
        print(f"[run ] {label}: {cfg}", flush=True)
        fs, bp, extra = evaluate_config(spec, X, y, cfg, quick=args.quick, memory=memory)
        if memory is not None:
            memory.clear(warn=False)
        log_run(f"combined:{label}",
                tags={"stage": "combined", "mode": mode, "group": "combined", "level": label, "winner": wname},
                params={**cfg, "cfg_json": json.dumps(cfg), "winner": wname},
                fold_scores=fs, best_params=bp, extra=extra)
        fold_store[label] = fs[PRIMARY]
        diff = fs[PRIMARY] - ref_scores[PRIMARY].to_numpy()
        test = corrected_resampled_ttest(diff, ratio)
        results[label] = {"mcc": float(np.nanmean(fs[PRIMARY])), "cfg": cfg,
                          "mean_diff_vs_baseline": test["mean_diff"], "t": test["t"], "p": test["p"],
                          "sensitivity": float(np.nanmean(fs["sensitivity"])),
                          "specificity": float(np.nanmean(fs["specificity"]))}
        print(f"        mcc = {results[label]['mcc']:.3f} | selisih vs baseline = {test['mean_diff']:+.3f}"
              f" | p (terkoreksi) = {test['p']:.3f}")

    # ---- Ablasi leave-one-out (hanya untuk pembahasan; tidak mempengaruhi pilihan final) ----
    loo = {}
    if "combined" in candidates and not args.no_loo:
        group_keys = {"scaler": ["scaler"], "encoding": ["encoding"], "imbalance": ["resample", "threshold"],
                      "feature_engineering": ["fe"], "feature_selection": ["fs"]}
        adopted = [g for g, v in sel["chosen_per_group"].items() if "cfg" in v]
        print("\n--- Ablasi leave-one-out pada konfigurasi gabungan (untuk pembahasan) ---")
        for g in adopted:
            variant = dict(candidates["combined"])
            for k in group_keys[g]:
                variant[k] = BASE_CFG[k]
            print(f"[run ] tanpa {g}: {variant}", flush=True)
            fs, bp, extra = evaluate_config(spec, X, y, variant, quick=args.quick, memory=memory)
            if memory is not None:
                memory.clear(warn=False)
            log_run(f"combined_ablation:without_{g}",
                    tags={"stage": "combined_ablation", "mode": mode, "group": g, "level": f"without_{g}", "winner": wname},
                    params={**variant, "cfg_json": json.dumps(variant), "winner": wname},
                    fold_scores=fs, best_params=bp, extra=extra)
            d = fs[PRIMARY] - fold_store["combined"]
            t = corrected_resampled_ttest(d, ratio)
            loo[g] = {"mcc_without": float(np.nanmean(fs[PRIMARY])), "delta_vs_combined": t["mean_diff"], "p": t["p"]}
            print(f"        mcc tanpa komponen = {loo[g]['mcc_without']:.3f} | selisih vs gabungan = {t['mean_diff']:+.3f}"
                  f" (negatif = komponen membantu) | p = {t['p']:.3f}")

    final_label = max(results, key=lambda k: results[k]["mcc"])
    out = {"mode": mode, "winner_model": wname, "final": final_label,
           "final_cfg": results[final_label]["cfg"], "comparison": results, "loo_ablation": loo}
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2, default=str)
    print(f"\n>>> KONFIGURASI FINAL: {final_label}  (MCC {results[final_label]['mcc']:.3f} vs baseline {results['baseline']['mcc']:.3f})")
    print(f"    {results[final_label]['cfg']}")
    print(f"Disimpan: {args.out}\nBerikutnya: python -m experiments.run_final" + (" --quick" if args.quick else ""))


if __name__ == "__main__":
    main()
