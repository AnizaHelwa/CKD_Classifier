"""
Memilih model terbaik dari hasil baseline secara otomatis dan tercatat.

Aturan pemilihan (ditetapkan SEBELUM melihat hasil):
  1. Kandidat yang boleh menang: family == 'SVM' (SVC/NuSVC/LinearSVC x kernel).
     Dummy / Logistic Regression / Random Forest hanya sebagai konteks pembanding.
  2. Pemenang = rata-rata metrik PRIMARY (MCC) tertinggi di outer CV.
  3. Kandidat yang selisihnya <= 1 standard error dari pemenang ditandai 'setara secara statistik'
     (informasi untuk pembahasan; bukan untuk mengganti pemenang).
  4. Pemenang harus melampaui Dummy; jika tidak, baseline dianggap tidak valid.

Jalankan:  python -m experiments.select_baseline_winner [--mode full|quick]
"""
import argparse
import json
import os
from collections import Counter

import mlflow
import pandas as pd

from src.metrics import PRIMARY

EXPERIMENT = "CKD_Baseline_v2"
SHOW = ["mcc", "balanced_acc", "f1_macro", "sensitivity", "specificity", "roc_auc", "pr_auc_nonckd"]


def modal_params(run_id):
    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="best_params_per_fold.json")
    with open(path, encoding="utf-8") as f:
        plist = json.load(f)
    c = Counter(json.dumps(p, sort_keys=True) for p in plist)
    best, n = c.most_common(1)[0]
    return json.loads(best), f"{n}/{len(plist)} fold"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full", choices=["full", "quick"])
    ap.add_argument("--out", default="results/baseline_winner.json")
    args = ap.parse_args()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT],
        filter_string=f"tags.stage = 'baseline' and tags.mode = '{args.mode}' "
                      f"and attributes.status = 'FINISHED'")
    if runs.empty:
        raise SystemExit(f"Tidak ada run baseline mode '{args.mode}'. Jalankan run_baseline dulu.")

    df = pd.DataFrame({"name": runs["tags.mlflow.runName"], "run_id": runs["run_id"],
                       "family": runs["tags.family"], "svm_type": runs["params.svm_type"],
                       "kernel": runs["params.kernel"]})
    for m in SHOW:
        df[m] = runs[f"metrics.{m}_mean"]
        df[f"{m}_se"] = runs[f"metrics.{m}_se"]
    df["n_failed_folds"] = runs["metrics.n_failed_folds"]
    df = df.sort_values(PRIMARY, ascending=False).reset_index(drop=True)

    svm = df[df.family == "SVM"].reset_index(drop=True)
    if svm.empty:
        raise SystemExit("Belum ada kandidat SVM yang selesai.")
    win = svm.iloc[0]
    svm["within_1se_of_winner"] = svm[PRIMARY] >= win[PRIMARY] - win[f"{PRIMARY}_se"]

    dummy = df[df.name == "dummy_most_frequent"]
    beats_dummy = bool(dummy.empty or win[PRIMARY] > float(dummy.iloc[0][PRIMARY]))

    print(f"\n=== Peringkat SEMUA kandidat (metrik utama: {PRIMARY}) | mode={args.mode} ===")
    show = df[["name", "family"] + SHOW].copy()
    print(show.round(3).to_string(index=False))
    print(f"\n=== Kandidat SVM (yang boleh menang) ===")
    print(svm[["name", PRIMARY, f"{PRIMARY}_se", "within_1se_of_winner"]].round(3).to_string(index=False))

    params, freq = modal_params(win["run_id"])
    print(f"\n>>> PEMENANG: {win['name']}  ({PRIMARY}={win[PRIMARY]:.3f})")
    print(f"    hyperparameter paling sering terpilih: {params}  [{freq}]")
    print(f"    setara secara statistik (<=1 SE): {svm[svm.within_1se_of_winner].name.tolist()}")
    print(f"    melampaui Dummy: {beats_dummy}")
    if not beats_dummy:
        print("    PERINGATAN: pemenang tidak melampaui Dummy -> baseline tidak valid.")

    out = {
        "protocol": "v2", "mode": args.mode, "primary_metric": PRIMARY,
        "winner": {"name": win["name"], "svm_type": win["svm_type"], "kernel": win["kernel"],
                   "metrics": {m: float(win[m]) for m in SHOW},
                   "modal_best_params": params, "modal_params_frequency": freq},
        "within_1se": svm[svm.within_1se_of_winner].name.tolist(),
        "beats_dummy": beats_dummy,
        "ranking_svm": svm[["name", PRIMARY]].round(4).to_dict("records"),
        "context": {r["name"]: round(float(r[PRIMARY]), 4)
                    for _, r in df[df.family != "SVM"].iterrows()},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.drop(columns=["run_id"]).to_csv(args.out.replace(".json", "_ranking.csv"), index=False)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nDisimpan: {args.out}  (dibaca oleh tahap berikutnya lewat src.winner.load_winner)")


if __name__ == "__main__":
    main()
