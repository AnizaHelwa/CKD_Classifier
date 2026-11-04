"""
Tahap 2d - Tuning akhir (grid lebih halus) + evaluasi final.

Secara default HANYA tuning di data latih (test set tidak dibuka). Test set baru dibuka dengan
--evaluate-test, dan hanya BOLEH SEKALI (dijaga oleh results/final_test_done.flag).

    python -m experiments.run_final                    # tuning + estimasi CV (nested), tanpa test set
    python -m experiments.run_final --evaluate-test    # + evaluasi final di test set (sekali saja)
"""
import argparse
import json
import os
import tempfile
import warnings

import joblib
import mlflow
import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.baseline import build_candidates
from src.data import CV_SEED, load_split
from src.metrics import PRIMARY, SCORING
from src.stage2 import build_pipeline, evaluate_config, final_grid, fitted_pipeline, hit_max_iter, make_search
from src.tracking import log_run

warnings.filterwarnings("ignore")
EXPERIMENT = "CKD_Stage2_v2"
FLAG = "results/final_test_done.flag"


def _score(model, X):
    return model.decision_function(X) if hasattr(model, "decision_function") else model.predict_proba(X)[:, 1]


def test_metrics(y, pred, s):
    y = np.asarray(y)
    return {"mcc": matthews_corrcoef(y, pred), "balanced_acc": balanced_accuracy_score(y, pred),
            "f1_macro": f1_score(y, pred, average="macro"),
            "sensitivity": recall_score(y, pred, pos_label=1), "specificity": recall_score(y, pred, pos_label=0),
            "roc_auc": roc_auc_score(y, s), "pr_auc_nonckd": average_precision_score((y == 0).astype(int), -s)}


def bootstrap_ci(y, pred, s, n=2000, seed=CV_SEED):
    """CI persentil 95% dengan bootstrap stratif. Penting: kelas non-CKD di test set hanya puluhan sampel."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
    rows = []
    for _ in range(n):
        ii = np.r_[rng.choice(idx0, len(idx0)), rng.choice(idx1, len(idx1))]
        rows.append(test_metrics(y[ii], pred[ii], s[ii]))
    d = pd.DataFrame(rows)
    return {k: [float(d[k].quantile(0.025)), float(d[k].quantile(0.975))] for k in d.columns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/CKD.csv")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cache", action="store_true", help="aktifkan cache joblib (lebih cepat di Linux/Mac; di Windows bisa PermissionError)")
    ap.add_argument("--evaluate-test", action="store_true")
    ap.add_argument("--force", action="store_true", help="izinkan membuka test set lagi (merusak keabsahan hasil)")
    ap.add_argument("--config", default="results/stage2_final_config.json")
    args = ap.parse_args()
    mode = "quick" if args.quick else "full"

    final = json.load(open(args.config, encoding="utf-8"))
    if final["mode"] != mode:
        raise SystemExit(f"stage2_final_config.json bermode '{final['mode']}', run ini '{mode}'.")
    if args.evaluate_test and args.quick and not args.force:
        raise SystemExit("Evaluasi test set hanya boleh di mode penuh. Mode --quick tidak dipakai untuk membuka "
                         "test set (hasilnya kasar, dan test set cukup dibuka sekali).")
    if args.evaluate_test and os.path.exists(FLAG) and not args.force and mode == "full":
        raise SystemExit("Test set SUDAH pernah dibuka (results/final_test_done.flag). "
                         "Membukanya lagi merusak keabsahan hasil. Gunakan --force hanya jika Anda paham risikonya.")

    cfg, wname = final["final_cfg"], final["winner_model"]
    X_train, X_test, y_train, y_test = load_split(args.data)
    spec = build_candidates(y_train, quick=args.quick)[wname]
    grid = final_grid(spec, quick=args.quick)
    memory = Memory(location=".cache/pipeline", verbose=0) if args.cache else None
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT)
    print(f"Model: {wname} | konfigurasi final: {cfg}\n")

    print("[1/3] Estimasi CV (nested) untuk prosedur tuning akhir ...", flush=True)
    fs, bp, extra = evaluate_config(spec, X_train, y_train, cfg, grid=grid, quick=args.quick, memory=memory)
    if memory is not None:
        memory.clear(warn=False)
    log_run("final:nested_cv", tags={"stage": "final_cv", "mode": mode, "group": "final", "level": "tuned", "winner": wname},
            params={**cfg, "cfg_json": json.dumps(cfg), "winner": wname, "grid": json.dumps(grid, default=str)},
            fold_scores=fs, best_params=bp, extra=extra)
    print(f"      MCC (CV) = {np.nanmean(fs[PRIMARY]):.3f} ± {np.nanstd(fs[PRIMARY]):.3f}")
    nc = int(extra["n_nonconverged_folds"])
    print(f"      fold dengan solver tidak konvergen: {nc} dari {len(bp)}" + ("   <-- PERHATIAN" if nc else "   (aman)"))
    for m in ["balanced_acc", "sensitivity", "specificity", "roc_auc", "pr_auc_nonckd"]:
        print(f"      {m:14} = {np.nanmean(fs[m]):.3f} ± {np.nanstd(fs[m]):.3f}")

    print("\n[2/3] Fit model final di SELURUH data latih (tuning 5-fold) ...", flush=True)
    pipe = build_pipeline(spec["estimator"], X_train, cfg, memory=None)
    search = make_search(pipe, grid, cfg, StratifiedKFold(5, shuffle=True, random_state=CV_SEED))
    search.fit(X_train, y_train)
    print(f"      parameter terbaik: {search.best_params_}")
    clf = fitted_pipeline(search).named_steps["clf"]
    thr = getattr(search.best_estimator_, "best_threshold_", None)
    print(f"      n_iter_ = {getattr(clf, 'n_iter_', '-')} | max_iter = {getattr(clf, 'max_iter', '-')}"
          f" | threshold = {None if thr is None else round(float(thr), 4)}")
    if hit_max_iter(clf):
        print("      PERHATIAN: model final TIDAK konvergen (n_iter_ mencapai max_iter). Jangan buka test set dulu.")
    os.makedirs("results", exist_ok=True)
    joblib.dump(search.best_estimator_, f"results/final_model_{mode}.joblib")

    if not args.evaluate_test:
        print("\n[3/3] Test set TIDAK dibuka (tanpa --evaluate-test). Selesai.")
        return

    print("\n[3/3] Evaluasi final di TEST SET (sekali saja) ...", flush=True)
    pred, s = search.predict(X_test), _score(search, X_test)
    met, ci = test_metrics(y_test, pred, s), bootstrap_ci(y_test, pred, s, n=300 if args.quick else 2000)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    print(f"      n_test = {len(y_test)} (non-CKD = {int((np.asarray(y_test) == 0).sum())})")
    for k, v in met.items():
        print(f"      {k:14} = {v:.3f}   CI95% [{ci[k][0]:.3f}, {ci[k][1]:.3f}]")
    print("\n      Confusion matrix (baris = aktual, kolom = prediksi; urutan 0=non-CKD, 1=CKD):")
    print(f"      {cm.tolist()}\n")
    print(classification_report(y_test, pred, target_names=["non-CKD (0)", "CKD (1)"], digits=3))

    out = {"model": wname, "cfg": cfg, "best_params": search.best_params_, "test_metrics": met,
           "test_ci95": ci, "confusion_matrix": cm.tolist(),
           "cv_mcc_mean": float(np.nanmean(fs[PRIMARY]))}
    json.dump(out, open(f"results/final_test_{mode}.json", "w", encoding="utf-8"), indent=2, default=str)
    if mode == "full":
        open(FLAG, "w").write("test set dibuka sekali\n")
    with mlflow.start_run(run_name=f"final:test_{mode}"):
        mlflow.set_tags({"stage": "final_test", "mode": mode, "winner": wname})
        mlflow.log_params({k: str(v)[:500] for k, v in {**cfg, "best_params": search.best_params_}.items()})
        for k, v in met.items():
            mlflow.log_metric(f"test_{k}", float(v))
        with tempfile.TemporaryDirectory() as d:
            pd.DataFrame(cm, index=["aktual_0", "aktual_1"], columns=["pred_0", "pred_1"]).to_csv(os.path.join(d, "confusion_matrix.csv"))
            mlflow.log_artifacts(d)
    print("Disimpan: results/final_test_%s.json" % mode)


if __name__ == "__main__":
    main()
