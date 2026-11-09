"""
Membuat gambar-gambar laporan dari hasil yang SUDAH tersimpan (MLflow, results/*.json, model final).
Tidak melatih ulang apa pun. Setiap gambar dibuat terpisah; jika satu gagal, yang lain tetap jalan.

    python -m experiments.make_figures                    # semua gambar, mode full
    python -m experiments.make_figures --no-test-curves   # lewati gambar yang memakai data uji (ROC/PR/skor)

Keluaran: results/figures/*.png (300 dpi)

Catatan: gambar ROC/PR/skor dihitung ulang dari model final yang sudah tetap pada data uji. Ini hanya
deskriptif; tidak ada keputusan model yang boleh diubah setelah melihatnya.
"""
import argparse
import json
import os
import warnings

import matplotlib

matplotlib.use("Agg")                      # tanpa jendela GUI (aman di Windows)
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

from src.data import load_split
from src.tracking import load_fold_scores

warnings.filterwarnings("ignore")
EXP_BASE, EXP_S2 = "CKD_Baseline_v2", "CKD_Stage2_v2"
OUT = "results/figures"
C_FAM = {"SVM": "#1f77b4", "comparator": "#ff7f0e", "reference": "#7f7f7f"}
C_GOOD, C_BAD, C_NEUT = "#2ca02c", "#d62728", "#9e9e9e"
METRICS = [("mcc", "MCC"), ("balanced_acc", "Balanced acc."), ("f1_macro", "F1-macro"),
           ("sensitivity", "Sensitivity"), ("specificity", "Specificity"),
           ("roc_auc", "ROC-AUC"), ("pr_auc_nonckd", "PR-AUC\nnon-CKD")]
GROUP_ID = {"scaler": "Normalisasi", "encoding": "Encoding kategorikal",
            "imbalance": "Penanganan imbalance", "feature_engineering": "Feature engineering",
            "feature_selection": "Seleksi fitur"}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})


# ------------------------------------------------------------------ util
def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".png")
    fig.savefig(path)
    plt.close(fig)
    print("  ->", path)


def runs(exp, stage, mode):
    r = mlflow.search_runs(experiment_names=[exp], filter_string=(
        f"tags.stage = '{stage}' and tags.mode = '{mode}' and attributes.status = 'FINISHED'"))
    if r.empty:
        raise RuntimeError(f"tidak ada run stage='{stage}' mode='{mode}' di {exp}")
    r = r.copy()
    r["name"] = r["tags.mlflow.runName"]
    return r.sort_values("start_time")


def latest(df, name):
    d = df[df.name == name]
    return None if d.empty else d.iloc[-1]


def read_json(path):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


# ------------------------------------------------------------------ data
def fig_data(data):
    _, _, y_tr, y_te = load_split(data)
    parts = {"Seluruh data": pd.concat([y_tr, y_te]), "Data latih": y_tr, "Data uji": y_te}
    n0 = [int((v == 0).sum()) for v in parts.values()]
    n1 = [int((v == 1).sum()) for v in parts.values()]
    x, w = np.arange(3), 0.38
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for off, vals, col, lab in [(-w / 2, n0, C_BAD, "non-CKD (0)"), (w / 2, n1, C_FAM["SVM"], "CKD (1)")]:
        bars = ax.bar(x + off, vals, w, color=col, label=lab)
        for b, v, t in zip(bars, vals, np.array(n0) + np.array(n1)):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({v / t:.1%})", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(parts.keys())
    ax.set_ylabel("Jumlah pasien")
    ax.set_ylim(0, max(n1) * 1.18)
    ax.set_title("Distribusi kelas dan pembagian data")
    ax.legend(frameon=False)
    save(fig, "01_distribusi_kelas")


# ------------------------------------------------------------------ baseline
def fig_baseline_rank(mode, winner):
    b = runs(EXP_BASE, "baseline", mode).sort_values("metrics.mcc_mean").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(range(len(b)), b["metrics.mcc_mean"], xerr=b["metrics.mcc_se"],
            color=[C_FAM.get(f, C_NEUT) for f in b["tags.family"]], capsize=3)
    ax.set_yticks(range(len(b)))
    ax.set_yticklabels(b.name)
    for t in ax.get_yticklabels():
        if t.get_text() == winner:
            t.set_fontweight("bold")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("MCC rata-rata (nested CV) ± 1 SE")
    ax.set_title("Baseline: peringkat semua kandidat")
    ax.legend(handles=[Patch(color=C_FAM["SVM"], label="Varian SVM"), Patch(color=C_FAM["comparator"], label="Pembanding"),
                       Patch(color=C_FAM["reference"], label="Dummy (acuan)")], frameon=False, loc="lower right")
    save(fig, "02_baseline_peringkat_mcc")


def fig_baseline_heatmap(mode):
    b = runs(EXP_BASE, "baseline", mode).sort_values("metrics.mcc_mean", ascending=False).reset_index(drop=True)
    M = np.array([[b[f"metrics.{k}_mean"].iloc[i] for k, _ in METRICS] for i in range(len(b))])
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    im = ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([l for _, l in METRICS])
    ax.set_yticks(range(len(b)))
    ax.set_yticklabels(b.name)
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title("Baseline: seluruh metrik (diurutkan menurut MCC)")
    save(fig, "03_baseline_heatmap_metrik")


def fig_baseline_tradeoff(mode):
    b = runs(EXP_BASE, "baseline", mode).sort_values("metrics.sensitivity_mean", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    style = {"reference": ("s", 110, 1), "comparator": ("^", 60, 2), "SVM": ("o", 55, 3)}
    ys = np.linspace(0.96, 0.04, len(b))
    for i, r in b.iterrows():
        fam = r["tags.family"]
        mk, sz, z = style.get(fam, ("o", 55, 3))
        xy = (r["metrics.specificity_mean"], r["metrics.sensitivity_mean"])
        ax.scatter(*xy, s=sz, marker=mk, color=C_FAM.get(fam, C_NEUT), zorder=z)
        ax.annotate(r["name"], xy=xy, xytext=(1.04, ys[i]), textcoords="axes fraction", fontsize=7.5, va="center",
                    arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.6, shrinkA=0, shrinkB=3))
    ax.set_xlabel("Specificity (non-CKD terdeteksi)")
    ax.set_ylabel("Sensitivity (CKD terdeteksi)")
    ax.set_title("Baseline: hampir semua model condong menebak CKD")
    ax.grid(alpha=0.25)
    ax.legend(handles=[Patch(color=C_FAM["SVM"], label="Varian SVM"), Patch(color=C_FAM["comparator"], label="Pembanding"),
                       Patch(color=C_FAM["reference"], label="Dummy (acuan)")], frameon=False, loc="lower left")
    save(fig, "04_baseline_sensitivity_vs_specificity")


def fig_baseline_folds(mode):
    b = runs(EXP_BASE, "baseline", mode)
    data = {r["name"]: load_fold_scores(r["run_id"])["mcc"].to_numpy() for _, r in b.iterrows()}
    order = sorted(data, key=lambda k: np.nanmedian(data[k]))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.boxplot([data[k] for k in order], vert=False, labels=order, showfliers=False,
               boxprops=dict(color="#555"), medianprops=dict(color=C_BAD))
    rng = np.random.default_rng(0)
    for i, k in enumerate(order, start=1):
        ax.scatter(data[k], i + rng.normal(0, 0.06, len(data[k])), s=10, color=C_FAM["SVM"], alpha=0.6, zorder=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("MCC per fold (outer CV)")
    ax.set_title("Baseline: sebaran MCC antar fold")
    save(fig, "05_baseline_sebaran_fold")


# ------------------------------------------------------------------ tahap 2
def fig_screening(mode, sel):
    s = runs(EXP_S2, "screening", mode)
    ref = latest(s, "reference:baseline")
    r_mcc, r_se = ref["metrics.mcc_mean"], ref["metrics.mcc_se"]
    groups = [g for g in GROUP_ID if (s["tags.group"] == g).any()]
    heights = [(s["tags.group"] == g).sum() + 1.5 for g in groups]
    fig, axes = plt.subplots(len(groups), 1, figsize=(7.5, 0.3 * sum(heights) + 1.2), sharex=True,
                             gridspec_kw={"height_ratios": heights})
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, groups):
        d = s[s["tags.group"] == g].sort_values("metrics.mcc_mean").reset_index(drop=True)
        adopted = ((sel or {}).get("chosen_per_group", {}).get(g, {}) or {}).get("level")
        ax.axvspan(-r_se, r_se, color="#eeeeee", zorder=0)
        ax.axvline(0, color="k", lw=0.8)
        for i, r in d.iterrows():
            col = C_GOOD if r["tags.level"] == adopted else C_NEUT
            ax.errorbar(r["metrics.mcc_mean"] - r_mcc, i, xerr=r["metrics.mcc_se"], fmt="o", color=col,
                        capsize=2, ms=5)
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels([("tanpa scaling" if (g == "scaler" and lv == "none") else lv) for lv in d["tags.level"]], fontsize=8)
        ax.set_title(GROUP_ID[g], loc="left", fontsize=9, fontweight="bold", pad=3)
        ax.set_ylim(-0.7, len(d) - 0.3)
    axes[-1].set_xlabel("Selisih MCC terhadap baseline (garis nol); batang = ±1 SE")
    fig.suptitle("Screening komponen: hijau = level yang dipilih; pita abu = ±1 SE baseline", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "06_screening_komponen")


def fig_fs_curve(mode):
    s = runs(EXP_S2, "screening", mode)
    ref = latest(s, "reference:baseline")
    d = s[s["tags.group"] == "feature_selection"].copy()
    d["x"], d["y"], d["e"] = d["metrics.n_features_final"], d["metrics.mcc_mean"], d["metrics.mcc_se"]
    meth = {"anova": ("ANOVA", "o-"), "mi": ("Mutual Information", "s-"), "rfe": ("RFE", "^-"),
            "l1": ("L1 (berbagai C)", "D:"), "corr": ("Filter korelasi", "P")}
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for key, (lab, fmt) in meth.items():
        m = d[d["tags.level"].str.startswith(key)].sort_values("x")
        if len(m):
            ax.errorbar(m.x, m.y, yerr=m.e, fmt=fmt, capsize=2, label=lab, ms=5)
    ax.errorbar([ref["metrics.n_features_final"]], [ref["metrics.mcc_mean"]], yerr=[ref["metrics.mcc_se"]],
                fmt="k*", ms=11, label="Baseline (semua fitur)")
    ax.axhline(ref["metrics.mcc_mean"], color="k", ls="--", lw=0.8)
    ax.set_xlabel("Jumlah fitur yang masuk ke model")
    ax.set_ylabel("MCC (± 1 SE)")
    ax.set_title("Seleksi fitur: MCC vs jumlah fitur")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    save(fig, "07_seleksi_fitur_kurva")


def fig_combined(mode, fin):
    s, c = runs(EXP_S2, "screening", mode), runs(EXP_S2, "combined", mode)
    try:
        f = runs(EXP_S2, "final_cv", mode)
    except RuntimeError:
        f = None
    items = [("Baseline", latest(s, "reference:baseline"), None),
             ("Komponen tunggal\nterbaik", latest(c, "combined:best_single"), "best_single"),
             ("Gabungan", latest(c, "combined:combined"), "combined"),
             ("Final\n(tuning akhir)", None if f is None else latest(f, "final:nested_cv"), None)]
    items = [it for it in items if it[1] is not None]
    comp = (fin or {}).get("comparison", {})
    loo = (fin or {}).get("loo_ablation", {})
    fig, axes = plt.subplots(1, 2 if loo else 1, figsize=(11 if loo else 6, 4.2), squeeze=False)
    ax = axes[0, 0]
    for i, (lab, r, key) in enumerate(items):
        ax.bar(i, r["metrics.mcc_mean"], yerr=r["metrics.mcc_se"], capsize=4,
               color=C_NEUT if lab == "Baseline" else C_FAM["SVM"])
        txt = f"{r['metrics.mcc_mean']:.3f}"
        p = comp.get(key, {}).get("p") if key else None
        if p is not None and not (isinstance(p, float) and np.isnan(p)):
            txt += f"\np={p:.3f}"
        ax.text(i, r["metrics.mcc_mean"] + r["metrics.mcc_se"] + 0.008, txt, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([it[0] for it in items], fontsize=8)
    ax.set_ylabel("MCC (nested CV) ± 1 SE")
    ax.set_ylim(0, max(r["metrics.mcc_mean"] + r["metrics.mcc_se"] for _, r, _ in items) * 1.28)
    ax.set_title("Baseline vs konfigurasi hasil pengembangan\n(p: corrected resampled t-test vs baseline)", fontsize=10)
    if loo:
        ax = axes[0, 1]
        names = list(loo)
        contrib = [-loo[g]["delta_vs_combined"] for g in names]
        ax.barh(range(len(names)), contrib, color=[C_GOOD if v > 0 else C_BAD for v in contrib])
        hi_, lo_ = max(contrib + [0]), min(contrib + [0])
        span = (hi_ - lo_) or 0.1
        for i, g in enumerate(names):
            ax.text(hi_ + 0.06 * span, i, f"{contrib[i]:+.3f}  (p={loo[g]['p']:.2f})", va="center", ha="left", fontsize=8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([GROUP_ID.get(g, g) for g in names])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlim(lo_ - 0.08 * span, hi_ + 0.95 * span)
        ax.set_xlabel("Penurunan MCC bila komponen dilepas\n(positif = komponen membantu)")
        ax.set_title("Ablasi leave-one-out pada konfigurasi gabungan", fontsize=10)
    save(fig, "08_kombinasi_dan_ablasi")


# ------------------------------------------------------------------ hasil akhir
def fig_test(mode):
    res = read_json(f"results/final_test_{mode}.json")
    if res is None:
        raise RuntimeError(f"results/final_test_{mode}.json tidak ada (test set belum dibuka?)")
    cv = latest(runs(EXP_S2, "final_cv", mode), "final:nested_cv")
    cm = np.array(res["confusion_matrix"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.6})
    ax = axes[0]
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            frac = cm[i, j] / cm[i].sum()
            ax.text(j, i, f"{cm[i, j]}\n({frac:.0%} dari baris)", ha="center", va="center",
                    color="white" if frac > 0.5 else "black", fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["non-CKD", "CKD"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["non-CKD", "CKD"])
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title("Confusion matrix (data uji)")
    ax = axes[1]
    for i, (k, lab) in enumerate(METRICS):
        t, (lo, hi) = res["test_metrics"][k], res["test_ci95"][k]
        ax.errorbar(t, i - 0.15, xerr=[[t - lo], [hi - t]], fmt="o", color=C_GOOD, capsize=3,
                    label="Data uji (CI 95% bootstrap)" if i == 0 else None)
        if cv is not None:
            ax.errorbar(cv[f"metrics.{k}_mean"], i + 0.15, xerr=cv[f"metrics.{k}_std"], fmt="s", color=C_NEUT,
                        capsize=3, label="CV data latih (± 1 SD antar fold)" if i == 0 else None)
    ax.set_yticks(range(len(METRICS)))
    ax.set_yticklabels([l.replace("\n", " ") for _, l in METRICS])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    ax.set_title(f"Estimasi CV vs hasil uji (n non-CKD = {int(cm[0].sum())})")
    save(fig, "09_hasil_uji_dan_cv")


def _model(mode, data):
    import joblib
    model = joblib.load(f"results/final_model_{mode}.joblib")
    X_tr, X_te, y_tr, y_te = load_split(data)
    return model, X_tr, X_te, y_tr, np.asarray(y_te), read_json(f"results/final_test_{mode}.json")


def fig_curves(mode, data):
    model, _, X_te, _, y, res = _model(mode, data)
    s, cm = model.decision_function(X_te), np.array(res["confusion_matrix"])
    tm = res["test_metrics"]
    fpr, tpr, _ = roc_curve(y, s)
    p0, r0, _ = precision_recall_curve((y == 0).astype(int), -s)
    p1, r1, _ = precision_recall_curve(y, s)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    ax = axes[0]
    ax.plot(fpr, tpr, color=C_FAM["SVM"], label=f"AUC = {roc_auc_score(y, s):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Acak")
    ax.scatter([1 - tm["specificity"]], [tm["sensitivity"]], color=C_BAD, zorder=3, label="Titik operasi (threshold terpilih)")
    ax.set(xlabel="1 − Specificity", ylabel="Sensitivity", title="Kurva ROC (data uji)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax = axes[1]
    ax.plot(r0, p0, color=C_FAM["SVM"], label=f"AP = {average_precision_score((y == 0).astype(int), -s):.3f}")
    ax.axhline((y == 0).mean(), color="k", ls="--", lw=0.8, label=f"Acak (prevalensi {(y == 0).mean():.2f})")
    ax.scatter([tm["specificity"]], [cm[0][0] / max(cm[0][0] + cm[1][0], 1)], color=C_BAD, zorder=3, label="Titik operasi")
    ax.set(xlabel="Recall non-CKD", ylabel="Precision non-CKD", title="PR: kelas non-CKD (kelas langka)", ylim=(0, 1.02))
    ax.legend(frameon=False, fontsize=8)
    ax = axes[2]
    ax.plot(r1, p1, color=C_FAM["SVM"], label=f"AP = {average_precision_score(y, s):.3f}")
    ax.axhline((y == 1).mean(), color="k", ls="--", lw=0.8, label=f"Acak (prevalensi {(y == 1).mean():.2f})")
    ax.scatter([tm["sensitivity"]], [cm[1][1] / max(cm[0][1] + cm[1][1], 1)], color=C_BAD, zorder=3, label="Titik operasi")
    ax.set(xlabel="Recall CKD", ylabel="Precision CKD", title="PR: kelas CKD (mayoritas)", ylim=(max(0, (y == 1).mean() - 0.08), 1.005))
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, "10_kurva_roc_pr_uji")


def fig_scores(mode, data):
    model, _, X_te, _, y, _ = _model(mode, data)
    s, thr = model.decision_function(X_te), getattr(model, "best_threshold_", None)
    bins = np.linspace(s.min(), s.max(), 30)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(s[y == 1], bins=bins, density=True, alpha=0.55, color=C_FAM["SVM"], label=f"CKD (n={int((y == 1).sum())})")
    ax.hist(s[y == 0], bins=bins, density=True, alpha=0.65, color=C_BAD, label=f"non-CKD (n={int((y == 0).sum())})")
    if thr is not None:
        ax.axvline(thr, color="k", ls="--", label=f"Threshold terpilih = {float(thr):.2f}")
    ax.axvline(0, color="#888", ls=":", lw=1, label="Threshold bawaan SVM (0)")
    ax.set(xlabel="Skor keputusan model", ylabel="Kepadatan", title="Distribusi skor per kelas dan posisi threshold (data uji)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "11_distribusi_skor_threshold")


def fig_coef(mode, data):
    from src.baseline import _cat_cols, _cont_cols
    from src.features import add_features
    model, X_tr, *_ = _model(mode, data)
    pipe = getattr(model, "estimator_", model)
    steps = pipe.named_steps
    clf = steps["clf"]
    if not hasattr(clf, "coef_") or "ohe" in steps:
        raise RuntimeError("model tidak linear atau memakai one-hot: gambar koefisien dilewati")
    X = add_features(X_tr) if "fe" in steps else X_tr
    names = np.array(_cont_cols(X) + _cat_cols(X))
    if "fs" in steps:
        names = names[steps["fs"].get_support()]
    coef = clf.coef_.ravel()
    if len(names) != len(coef):
        raise RuntimeError("jumlah nama fitur tidak cocok dengan koefisien")
    nz = np.where(np.abs(coef) > 1e-12)[0]
    top = nz[np.argsort(np.abs(coef[nz]))[-15:]]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(top) + 1.6))
    ax.barh(range(len(top)), coef[top], color=[C_FAM["SVM"] if c > 0 else C_BAD for c in coef[top]])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(names[top], fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Koefisien LinearSVC (positif → mendorong prediksi CKD; negatif → non-CKD)")
    ax.set_title(f"{len(top)} fitur dengan koefisien terbesar (dari {len(nz)} fitur bukan nol)", fontsize=10)
    fig.text(0.01, -0.03, "Catatan: fitur kontinu distandardisasi, fitur kategorikal tidak; besarnya tidak sebanding langsung. "
             "Ini asosiasi model, bukan efek kausal.", fontsize=7, ha="left")
    save(fig, "12_koefisien_model")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full", choices=["full", "quick"])
    ap.add_argument("--data", default="data/CKD.csv")
    ap.add_argument("--no-test-curves", action="store_true", help="lewati gambar yang memakai data uji")
    args = ap.parse_args()
    mode = args.mode
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    winner = (read_json("results/baseline_winner.json") or {}).get("winner", {}).get("name")
    sel = read_json("results/stage2_selection.json")
    fin = read_json("results/stage2_final_config.json")

    tasks = [("distribusi kelas", lambda: fig_data(args.data)),
             ("baseline: peringkat", lambda: fig_baseline_rank(mode, winner)),
             ("baseline: heatmap", lambda: fig_baseline_heatmap(mode)),
             ("baseline: sensitivity vs specificity", lambda: fig_baseline_tradeoff(mode)),
             ("baseline: sebaran fold", lambda: fig_baseline_folds(mode)),
             ("screening", lambda: fig_screening(mode, sel)),
             ("seleksi fitur", lambda: fig_fs_curve(mode)),
             ("kombinasi + ablasi", lambda: fig_combined(mode, fin)),
             ("hasil uji + CV", lambda: fig_test(mode)),
             ("koefisien model", lambda: fig_coef(mode, args.data))]
    if not args.no_test_curves:
        tasks += [("kurva ROC/PR", lambda: fig_curves(mode, args.data)),
                  ("distribusi skor", lambda: fig_scores(mode, args.data))]

    ok = 0
    for name, fn in tasks:
        print(f"[{name}]")
        try:
            fn()
            ok += 1
        except Exception as e:                                   # satu gagal tidak menghentikan yang lain
            plt.close("all")
            print(f"  GAGAL: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(tasks)} gambar berhasil dibuat di {OUT}/")


if __name__ == "__main__":
    main()
