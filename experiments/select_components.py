"""
Tahap 2b - Memilih level terbaik per kelompok komponen dari hasil screening.

Aturan (ditetapkan sebelum melihat hasil data asli) - aturan 1-SE:
  * Di tiap kelompok, cari level dengan rata-rata MCC tertinggi (baseline ikut bersaing).
  * Jika baseline berada dalam 1 SE dari yang tertinggi, baseline DIPERTAHANKAN (komponen tidak
    dipakai): selisih sekecil itu tidak bisa dibedakan dari noise, dan memilihnya hanya menambah
    kerumitan / menumpuk kebetulan. Jika baseline di luar 1 SE, level tertinggi dipakai.
  * Alasan: memilih "tertinggi mutlak" di banyak kelompok sekaligus mengejar noise (winner's curse).
  * Konfigurasi gabungan = gabungan pilihan semua kelompok (diuji di tahap 2c).
  * Catatan: satu-faktor-sekali tidak menangkap interaksi antar komponen; karena itu gabungan
    divalidasi ulang dan dibandingkan dengan baseline serta dengan komponen tunggal terbaik.

    python -m experiments.select_components [--mode full|quick]
"""
import argparse
import json
import os

import mlflow
import pandas as pd

from src.metrics import PRIMARY
from src.stage2 import BASE_CFG

EXPERIMENT = "CKD_Stage2_v2"
SHOW = ["mcc", "balanced_acc", "sensitivity", "specificity", "roc_auc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full", choices=["full", "quick"])
    ap.add_argument("--out", default="results/stage2_selection.json")
    args = ap.parse_args()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT],
        filter_string=f"tags.stage = 'screening' and tags.mode = '{args.mode}' and attributes.status = 'FINISHED'")
    if runs.empty:
        raise SystemExit("Belum ada hasil screening. Jalankan run_screening dulu.")

    df = pd.DataFrame({"name": runs["tags.mlflow.runName"], "group": runs["tags.group"],
                       "level": runs["tags.level"], "cfg": runs["params.cfg_json"],
                       "n_feat": runs["metrics.n_features_final"]})
    for m in SHOW:
        df[m] = runs[f"metrics.{m}_mean"]
    df["se"] = runs[f"metrics.{PRIMARY}_se"]
    df["n_failed"] = runs["metrics.n_failed_folds"]
    ref = df[df.group == "reference"]
    if ref.empty:
        raise SystemExit("Run 'reference:baseline' tidak ditemukan.")
    ref = ref.iloc[0]

    chosen, singles = {}, []
    print(f"Referensi (baseline): MCC = {ref[PRIMARY]:.3f}  (SE {ref.se:.3f})\n")
    for g in ["scaler", "encoding", "imbalance", "feature_engineering", "feature_selection"]:
        sub = pd.concat([df[df.group == g], ref.to_frame().T]).copy()
        for c in SHOW + ["se"]:
            sub[c] = sub[c].astype(float)
        sub = sub.sort_values(PRIMARY, ascending=False).reset_index(drop=True)
        if sub.empty or len(sub) == 1:
            continue
        best = sub.iloc[0]
        base_mcc = float(ref[PRIMARY])
        baseline_within_1se = base_mcc >= float(best[PRIMARY]) - float(best.se)
        sub["delta_vs_baseline"] = sub[PRIMARY] - float(ref[PRIMARY])
        sub["setara_1SE"] = sub[PRIMARY] >= best[PRIMARY] - best.se
        print(f"=== {g} ===")
        print(sub[["level", PRIMARY, "delta_vs_baseline", "se", "sensitivity", "specificity",
                   "n_feat", "setara_1SE"]].round(3).to_string(index=False))
        adopt = (best.group != "reference") and not baseline_within_1se
        picked = best.level if adopt else "baseline"
        why = ("baseline sudah yang tertinggi" if best.group == "reference" else
               f"{best.level} tertinggi ({best[PRIMARY]:.3f}) tapi baseline masih dalam 1 SE -> tidak cukup bukti"
               if baseline_within_1se else f"baseline di luar 1 SE dari {best.level}")
        print(f"  -> dipilih: {picked}   ({why})\n")
        chosen[g] = {"level": picked, "top1_level": best.level, "top1_mcc": float(best[PRIMARY])}
        if adopt:
            chosen[g]["cfg"] = json.loads(best.cfg)
            singles.append((float(best[PRIMARY]), g, json.loads(best.cfg)))

    combined = dict(BASE_CFG)
    for g, v in chosen.items():
        if "cfg" in v:
            for k, val in v["cfg"].items():
                if val != BASE_CFG[k]:
                    combined[k] = val
    best_single = max(singles, key=lambda t: t[0]) if singles else None
    out = {"mode": args.mode, "primary_metric": PRIMARY, "reference_mcc": float(ref[PRIMARY]),
           "chosen_per_group": chosen, "combined_cfg": combined,
           "best_single": ({"group": best_single[1], "mcc": best_single[0], "cfg": best_single[2]}
                           if best_single else None)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print("Konfigurasi gabungan:", combined)
    print(f"Disimpan: {args.out}\nBerikutnya: python -m experiments.run_combined" + (" --quick" if args.mode == "quick" else ""))


if __name__ == "__main__":
    main()
