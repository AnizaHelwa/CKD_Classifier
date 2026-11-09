"""
Mengekspor semua angka laporan dari MLflow + results/*.json menjadi tabel siap pakai, sekaligus
cadangan data (karena mlflow.db tidak ikut Git).

    python -m experiments.export_tables                 # mode full
    python -m experiments.export_tables --decimal .     # desimal titik pada kolom teks "ringkas"
    python -m experiments.export_tables --sep ,         # pemisah CSV koma (default ';' untuk Excel Indonesia)

Keluaran:
  results/tabel_laporan.xlsx      semua tabel dalam satu berkas (butuh: pip install openpyxl)
  results/tables/*.csv            versi CSV tiap tabel (UTF-8 dengan BOM, terbaca Excel)
  results/tables/arsip_*.csv      cadangan mentah: semua run MLflow dan skor per fold

Setiap tabel punya dua versi: "ringkas" (teks 'rata-rata ± SD', untuk ditempel ke laporan) dan
"angka" (kolom numerik, untuk analisis lanjutan).
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import warnings
from collections import Counter

import mlflow
import numpy as np
import pandas as pd

from src.stage2 import screening_configs
from src.tracking import load_fold_scores

warnings.filterwarnings("ignore")
EXP_BASE, EXP_S2 = "CKD_Baseline_v2", "CKD_Stage2_v2"
OUT_DIR, XLSX = "results/tables", "results/tabel_laporan.xlsx"
GROUPS = ["scaler", "encoding", "imbalance", "feature_engineering", "feature_selection"]
GROUP_ID = {"scaler": "Normalisasi", "encoding": "Encoding kategorikal", "imbalance": "Penanganan imbalance",
            "feature_engineering": "Feature engineering", "feature_selection": "Seleksi fitur"}
FAMILY_ID = {"SVM": "Varian SVM", "comparator": "Pembanding", "reference": "Acuan (Dummy)"}
MET = [("mcc", "MCC"), ("balanced_acc", "Balanced accuracy"), ("f1_macro", "F1-macro"),
       ("sensitivity", "Sensitivity (CKD)"), ("specificity", "Specificity (non-CKD)"),
       ("roc_auc", "ROC-AUC"), ("pr_auc_nonckd", "PR-AUC non-CKD"), ("pr_auc_ckd", "PR-AUC CKD")]
DEC = ","


# ------------------------------------------------------------------ util
def num(x, nd=3):
    """Format angka untuk kolom teks; desimal sesuai --decimal. NaN -> kosong."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{x:.{nd}f}".replace(".", DEC)


def ms(mean, sd):
    return "" if mean is None or np.isnan(mean) else f"{num(mean)} ± {num(sd)}"


def val(r, key):
    v = r.get(f"metrics.{key}", np.nan)
    return float(v) if v is not None else np.nan


def fv(v):
    """Nilai konfigurasi yang terbaca manusia (bukan JSON mentah); desimal mengikuti --decimal."""
    if v is None:
        return "tidak ada"
    if isinstance(v, bool):
        return "ya" if v else "tidak"
    if isinstance(v, float):
        return f"{v:.4g}".replace(".", DEC)
    if isinstance(v, (list, tuple)):
        return " ".join(fv(x) for x in v)
    return str(v)


def read_json(path):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def fetch(exp, mode, stage):
    r = mlflow.search_runs(experiment_names=[exp], filter_string=(
        f"tags.mode = '{mode}' and tags.stage = '{stage}' and attributes.status = 'FINISHED'"))
    if r.empty:
        return r
    r = r.copy()
    r["name"] = r["tags.mlflow.runName"]
    return r.sort_values("start_time")


def latest(df, name):
    if df.empty:
        return None
    d = df[df.name == name]
    return None if d.empty else d.iloc[-1]


def modal_params(run_id):
    try:
        p = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="best_params_per_fold.json")
        lst = json.load(open(p, encoding="utf-8"))
        best, n = Counter(json.dumps(x, sort_keys=True) for x in lst).most_common(1)[0]
        d = json.loads(best)
        if not d:
            return "-"
        txt = ", ".join(f"{k.split('__')[-1]}={fv(v)}" for k, v in d.items())
        return f"{txt} ({n}/{len(lst)} fold)"
    except Exception:
        return ""


# ------------------------------------------------------------------ tabel
def t_baseline(mode, wj):
    b = fetch(EXP_BASE, mode, "baseline")
    if b.empty:
        raise RuntimeError("tidak ada run baseline")
    b = b.sort_values("metrics.mcc_mean", ascending=False)
    win = ((wj or {}).get("winner") or {}).get("name")
    within = set((wj or {}).get("within_1se", []))
    disp, numr = [], []
    for _, r in b.iterrows():
        base = {"Kandidat": r["name"], "Keluarga": FAMILY_ID.get(r["tags.family"], r["tags.family"]),
                "Tipe SVM": r["params.svm_type"], "Kernel": r["params.kernel"]}
        d, n = dict(base), dict(base)
        for k, lab in MET:
            d[lab] = ms(val(r, f"{k}_mean"), val(r, f"{k}_std"))
            n[f"{lab} mean"], n[f"{lab} SD"], n[f"{lab} SE"] = val(r, f"{k}_mean"), val(r, f"{k}_std"), val(r, f"{k}_se")
        d["Hiperparameter (modus antar fold)"] = modal_params(r["run_id"])
        d["Fold gagal"] = n["Fold gagal"] = int(val(r, "n_failed_folds"))
        d["Pemenang"] = "ya" if r["name"] == win else ""
        d["Setara (≤1 SE) dengan pemenang"] = "ya" if r["name"] in within else ""
        n["Pemenang"], n["Setara ≤1 SE"] = d["Pemenang"], d["Setara (≤1 SE) dengan pemenang"]
        disp.append(d)
        numr.append(n)
    return pd.DataFrame(disp), pd.DataFrame(numr), b


def t_screening(mode, sel):
    s = fetch(EXP_S2, mode, "screening")
    ref = latest(s, "reference:baseline")
    if ref is None:
        raise RuntimeError("run reference:baseline tidak ada")
    rm = val(ref, "mcc_mean")
    disp, numr = [], []

    def row(group, level, r, within, adopted):
        mcc, se = val(r, "mcc_mean"), val(r, "mcc_se")
        d = {"Kelompok": group, "Level": level, "MCC (rata-rata ± SD)": ms(mcc, val(r, "mcc_std")),
             "SE": num(se), "Selisih vs baseline": num(mcc - rm), "Sensitivity": num(val(r, "sensitivity_mean")),
             "Specificity": num(val(r, "specificity_mean")), "Jumlah fitur akhir": num(val(r, "n_features_final"), 1),
             "Fold tidak konvergen": num(val(r, "n_nonconverged_folds"), 0),
             "Setara (≤1 SE) dengan terbaik di kelompok": within, "Dipilih": adopted}
        n = {"Kelompok": group, "Level": level, "MCC mean": mcc, "MCC SD": val(r, "mcc_std"), "MCC SE": se,
             "Selisih vs baseline": mcc - rm, "Sensitivity": val(r, "sensitivity_mean"),
             "Specificity": val(r, "specificity_mean"), "Fitur akhir": val(r, "n_features_final"),
             "Threshold rata-rata": val(r, "threshold_mean"), "Fold tidak konvergen": val(r, "n_nonconverged_folds"),
             "Setara ≤1 SE": within, "Dipilih": adopted}
        disp.append(d)
        numr.append(n)

    row("Acuan", "baseline", ref, "", "")
    for g in GROUPS:
        d = s[s["tags.group"] == g]
        if d.empty:
            continue
        pool = [(r["tags.level"], val(r, "mcc_mean"), val(r, "mcc_se")) for _, r in d.iterrows()]
        pool.append(("baseline", rm, val(ref, "mcc_se")))
        _, bm, bse = max(pool, key=lambda t: t[1])
        chosen = (((sel or {}).get("chosen_per_group") or {}).get(g) or {}).get("level")
        for _, r in d.sort_values("metrics.mcc_mean", ascending=False).iterrows():
            lvl = r["tags.level"]
            row(GROUP_ID[g], lvl, r, "ya" if val(r, "mcc_mean") >= bm - bse else "", "ya" if lvl == chosen else "")
    return pd.DataFrame(disp), pd.DataFrame(numr), s


def t_combined(mode, fin):
    s, c, f = fetch(EXP_S2, mode, "screening"), fetch(EXP_S2, mode, "combined"), fetch(EXP_S2, mode, "final_cv")
    comp = (fin or {}).get("comparison", {})
    items = [("Baseline", latest(s, "reference:baseline"), "baseline"),
             ("Komponen tunggal terbaik", latest(c, "combined:best_single"), "best_single"),
             ("Gabungan", latest(c, "combined:combined"), "combined"),
             ("Final (tuning akhir, CV)", latest(f, "final:nested_cv"), None)]
    base_m = val(items[0][1], "mcc_mean")
    disp, numr = [], []
    for lab, r, key in items:
        if r is None:
            continue
        info = comp.get(key, {}) if key else {}
        cfg = info.get("cfg") or (json.loads(r["params.cfg_json"]) if r.get("params.cfg_json") else {})
        d = {"Konfigurasi": lab, "MCC (rata-rata ± SD)": ms(val(r, "mcc_mean"), val(r, "mcc_std")),
             "Selisih vs baseline": num(val(r, "mcc_mean") - base_m), "p (terkoreksi)": num(info.get("p")),
             "Sensitivity": num(val(r, "sensitivity_mean")), "Specificity": num(val(r, "specificity_mean")),
             "Komponen": "; ".join(f"{k}={fv(v)}" for k, v in cfg.items())}
        n = {"Konfigurasi": lab, "MCC mean": val(r, "mcc_mean"), "MCC SD": val(r, "mcc_std"), "MCC SE": val(r, "mcc_se"),
             "Selisih vs baseline": val(r, "mcc_mean") - base_m, "p terkoreksi": info.get("p", np.nan),
             "Sensitivity": val(r, "sensitivity_mean"), "Specificity": val(r, "specificity_mean"), "Komponen": d["Komponen"]}
        disp.append(d)
        numr.append(n)
    loo = (fin or {}).get("loo_ablation") or {}
    loo_rows = [{"Komponen yang dilepas": GROUP_ID.get(g, g), "MCC tanpa komponen": num(v["mcc_without"]),
                 "Kontribusi komponen (positif = membantu)": num(-v["delta_vs_combined"]), "p (terkoreksi)": num(v["p"])}
                for g, v in loo.items()]
    return pd.DataFrame(disp), pd.DataFrame(numr), pd.DataFrame(loo_rows)


def t_final(mode):
    res = read_json(f"results/final_test_{mode}.json")
    if res is None:
        raise RuntimeError(f"results/final_test_{mode}.json tidak ada (test set belum dibuka?)")
    cv = latest(fetch(EXP_S2, mode, "final_cv"), "final:nested_cv")
    disp, numr = [], []
    for k, lab in MET[:-1]:
        t, (lo, hi) = res["test_metrics"][k], res["test_ci95"][k]
        cvm, cvs = (val(cv, f"{k}_mean"), val(cv, f"{k}_std")) if cv is not None else (np.nan, np.nan)
        disp.append({"Metrik": lab, "Data uji": num(t), "CI 95% (bootstrap)": f"[{num(lo)}; {num(hi)}]",
                     "CV data latih (rata-rata ± SD)": ms(cvm, cvs)})
        numr.append({"Metrik": lab, "Data uji": t, "CI95 bawah": lo, "CI95 atas": hi, "CV mean": cvm, "CV SD": cvs})
    cm = np.array(res["confusion_matrix"])
    cm_df = pd.DataFrame(cm, index=["Aktual non-CKD (0)", "Aktual CKD (1)"], columns=["Prediksi non-CKD (0)", "Prediksi CKD (1)"])
    cm_df.insert(0, "Aktual \\ Prediksi", cm_df.index)
    per = []
    for i, name in enumerate(["non-CKD (0)", "CKD (1)"]):
        tp, sup, pred = cm[i, i], cm[i].sum(), cm[:, i].sum()
        p, r = (tp / pred if pred else np.nan), (tp / sup if sup else np.nan)
        per.append({"Kelas": name, "Precision": num(p), "Recall": num(r),
                    "F1": num(2 * p * r / (p + r)) if p == p and r == r and (p + r) > 0 else "", "Support": int(sup)})
    per.append({"Kelas": "Accuracy", "Precision": "", "Recall": "", "F1": num(np.trace(cm) / cm.sum()), "Support": int(cm.sum())})
    return pd.DataFrame(disp), pd.DataFrame(numr), cm_df.reset_index(drop=True), pd.DataFrame(per), res


def t_config(mode, res, fin):
    rows = []
    cfg = (fin or {}).get("final_cfg") or (res or {}).get("cfg") or {}
    rows += [{"Item": f"Komponen: {k}", "Nilai": fv(v)} for k, v in cfg.items()]
    rows += [{"Item": f"Hiperparameter: {k.split('__')[-1]}", "Nilai": fv(v)} for k, v in ((res or {}).get("best_params") or {}).items()]
    try:
        import joblib
        m = joblib.load(f"results/final_model_{mode}.joblib")
        rows.append({"Item": "Threshold keputusan (hasil tuning)", "Nilai": num(float(m.best_threshold_), 4)})
        clf = getattr(m, "estimator_", m).named_steps["clf"]
        rows.append({"Item": "Iterasi solver (n_iter_ / max_iter)", "Nilai": f"{getattr(clf, 'n_iter_', '-')} / {getattr(clf, 'max_iter', '-')}"})
    except Exception:
        pass
    return pd.DataFrame(rows)


def t_metadata(mode, wj):
    def git(*a):
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    import matplotlib, scipy, sklearn
    try:
        import imblearn
        ib = imblearn.__version__
    except Exception:
        ib = ""
    rows = [("Tanggal ekspor", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")), ("Mode", mode),
            ("Git commit", git("rev-parse", "--short", "HEAD")), ("Git branch", git("rev-parse", "--abbrev-ref", "HEAD")),
            ("Git tag terdekat", git("describe", "--tags", "--always")),
            ("Python", sys.version.split()[0]), ("Platform", platform.platform()),
            ("scikit-learn", sklearn.__version__), ("imbalanced-learn", ib), ("numpy", np.__version__),
            ("pandas", pd.__version__), ("scipy", scipy.__version__), ("mlflow", mlflow.__version__),
            ("matplotlib", matplotlib.__version__)]
    b = fetch(EXP_BASE, mode, "baseline")
    if not b.empty:
        r = b.iloc[-1]
        for k in ["data_sha256", "split_seed", "cv_seed", "n_train", "minority_frac", "preprocessing", "imbalance_handling",
                  "inner_cv", "outer_cv", "tuning_metric"]:
            rows.append((f"Protokol baseline: {k}", r.get(f"params.{k}", "")))
    if wj:
        rows.append(("Pemenang baseline", wj["winner"]["name"]))
        rows.append(("Metrik utama", wj.get("primary_metric", "")))
    return pd.DataFrame(rows, columns=["Item", "Nilai"])


# ------------------------------------------------------------------ arsip mentah
def dump_archive():
    allr = mlflow.search_runs(experiment_names=[EXP_BASE, EXP_S2], output_format="pandas")
    folds = []
    for _, r in allr.iterrows():
        try:
            f = load_fold_scores(r["run_id"]).reset_index()
        except Exception:
            continue
        f.insert(0, "run_name", r.get("tags.mlflow.runName"))
        f.insert(0, "experiment_id", r["experiment_id"])
        f.insert(2, "stage", r.get("tags.stage"))
        f.insert(3, "mode", r.get("tags.mode"))
        f.insert(4, "run_id", r["run_id"])
        folds.append(f)
    return allr, (pd.concat(folds, ignore_index=True) if folds else pd.DataFrame())


# ------------------------------------------------------------------ pemeriksaan konsistensi
def consistency(mode, wj, base_runs, scr_runs, res):
    out = []
    if wj and not base_runs.empty:
        r = latest(base_runs, wj["winner"]["name"])
        d = abs(val(r, "mcc_mean") - wj["winner"]["metrics"]["mcc"]) if r is not None else np.nan
        out.append(("MCC pemenang baseline: MLflow vs baseline_winner.json", d < 1e-6, f"selisih {d:.2e}"))
    if res:
        t = latest(fetch(EXP_S2, mode, "final_test"), f"final:test_{mode}")
        if t is not None:
            d = abs(val(t, "test_mcc") - res["test_metrics"]["mcc"])
            out.append(("MCC test: MLflow vs final_test json", d < 1e-6, f"selisih {d:.2e}"))
    expected = len(screening_configs(mode == "quick")) + 1
    ok = (len(scr_runs) == expected)
    out.append((f"Jumlah run screening = {expected} (1 acuan + konfigurasi)", ok, f"ditemukan {len(scr_runs)}"))
    out.append(("Jumlah kandidat baseline = 12", len(base_runs) == 12, f"ditemukan {len(base_runs)}"))
    return out


# ------------------------------------------------------------------ main
def main():
    global DEC
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full", choices=["full", "quick"])
    ap.add_argument("--decimal", default=",", choices=[",", "."], help="desimal pada kolom teks 'ringkas'")
    ap.add_argument("--sep", default=";", help="pemisah CSV (default ';' agar terbaca Excel Indonesia)")
    args = ap.parse_args()
    DEC, mode = args.decimal, args.mode
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    os.makedirs(OUT_DIR, exist_ok=True)

    wj, sel = read_json("results/baseline_winner.json"), read_json("results/stage2_selection.json")
    fin = read_json("results/stage2_final_config.json")
    sheets, errors = {}, []

    def add(name, df):
        sheets[name] = df

    def attempt(label, fn):
        try:
            return fn()
        except Exception as e:
            errors.append(f"{label}: {type(e).__name__}: {e}")
            return None

    base_runs = scr_runs = pd.DataFrame()
    r = attempt("baseline", lambda: t_baseline(mode, wj))
    if r:
        add("T1 Baseline", r[0]); add("T1 Baseline (angka)", r[1]); base_runs = r[2]
    r = attempt("screening", lambda: t_screening(mode, sel))
    if r:
        add("T2 Screening", r[0]); add("T2 Screening (angka)", r[1]); scr_runs = r[2]
    r = attempt("kombinasi", lambda: t_combined(mode, fin))
    if r:
        add("T3 Kombinasi", r[0]); add("T3 Kombinasi (angka)", r[1])
        if len(r[2]):
            add("T3 Ablasi leave-one-out", r[2])
    res = None
    r = attempt("hasil uji", lambda: t_final(mode))
    if r:
        add("T4 Hasil uji", r[0]); add("T4 Hasil uji (angka)", r[1]); add("T4 Confusion matrix", r[2])
        add("T4 Per kelas", r[3]); res = r[4]
    c = attempt("konfigurasi final", lambda: t_config(mode, res, fin))
    if c is not None:
        add("T5 Konfigurasi final", c)
    m = attempt("metadata", lambda: t_metadata(mode, wj))
    if m is not None:
        add("Metadata", m)
    arc = attempt("arsip", dump_archive)

    readme = pd.DataFrame([
        ("T1 Baseline", "12 kandidat, semua metrik (rata-rata ± SD), hiperparameter modus, pemenang", "MLflow: CKD_Baseline_v2"),
        ("T2 Screening", "Tiap level komponen vs baseline, level yang dipilih (aturan 1-SE)", "MLflow: CKD_Stage2_v2"),
        ("T3 Kombinasi", "Baseline vs komponen tunggal vs gabungan vs final, p terkoreksi", "MLflow + stage2_final_config.json"),
        ("T3 Ablasi leave-one-out", "Kontribusi tiap komponen dalam konfigurasi gabungan", "stage2_final_config.json"),
        ("T4 Hasil uji", "Metrik data uji dengan CI 95% bootstrap vs estimasi CV", "final_test json + MLflow"),
        ("T4 Confusion matrix / Per kelas", "Rincian prediksi data uji", "final_test json"),
        ("T5 Konfigurasi final", "Komponen, hiperparameter, threshold, iterasi solver", "json + model final"),
        ("Metadata", "Versi pustaka, commit Git, protokol (untuk reproduksibilitas)", "lingkungan + MLflow"),
        ("Versi '(angka)'", "Kolom numerik mentah untuk analisis; versi tanpa label = teks ringkas untuk laporan", "-"),
        ("Catatan", "SD = simpangan baku antar fold; SE = SD/√(jumlah fold); fold CV berulang saling berbagi data "
                    "sehingga SE dan p cenderung optimistis", "-")], columns=["Sheet", "Isi", "Sumber"])
    sheets = {"README": readme, **sheets}

    for name, df in sheets.items():                                  # CSV
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        df.to_csv(os.path.join(OUT_DIR, slug + ".csv"), sep=args.sep, index=False, encoding="utf-8-sig")
    if arc:
        arc[0].to_csv(os.path.join(OUT_DIR, "arsip_semua_run_mlflow.csv"), sep=args.sep, index=False, encoding="utf-8-sig")
        arc[1].to_csv(os.path.join(OUT_DIR, "arsip_skor_per_fold.csv"), sep=args.sep, index=False, encoding="utf-8-sig")

    try:                                                                # Excel
        with pd.ExcelWriter(XLSX, engine="openpyxl") as xw:
            for name, df in sheets.items():
                df.to_excel(xw, sheet_name=name[:31], index=False)
            for ws in xw.book.worksheets:
                ws.freeze_panes = "A2"
                for col in ws.columns:
                    w = max((len(str(c.value)) for c in col if c.value is not None), default=8)
                    ws.column_dimensions[col[0].column_letter].width = min(w + 2, 60)
        xlsx_ok = True
    except ImportError:
        xlsx_ok = False

    print(f"Tabel ({len(sheets)} lembar) -> {OUT_DIR}/*.csv")
    if arc:
        print(f"Arsip mentah: {len(arc[0])} run, {len(arc[1])} baris skor per fold -> {OUT_DIR}/arsip_*.csv")
    print(f"Excel -> {XLSX}" if xlsx_ok else "Excel dilewati: jalankan 'pip install openpyxl' lalu ulangi.")
    print("\nPemeriksaan konsistensi:")
    for label, ok, info in consistency(mode, wj, base_runs, scr_runs, res):
        print(f"  [{'OK' if ok else 'PERIKSA'}] {label} ({info})")
    for e in errors:
        print("  GAGAL:", e)


if __name__ == "__main__":
    main()
