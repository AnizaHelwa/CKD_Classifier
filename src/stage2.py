"""
Tahap 2: pengembangan pemenang baseline.

Semua langkah yang belajar dari data (scaler, SMOTENC, seleksi fitur, threshold, model)
ada DI DALAM Pipeline, sehingga di-fit ulang di setiap fold -> tidak ada data leakage.

Urutan langkah:  [FE] -> scaling kontinu -> [SMOTENC] -> [one-hot nominal] -> [seleksi fitur] -> model
(scaling harus sebelum SMOTENC karena SMOTENC memakai jarak Euclidean pada fitur kontinu;
 one-hot setelah SMOTENC agar sampel sintetis tidak menghasilkan baris one-hot yang rusak.)
"""
from functools import partial

import numpy as np
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import (RFE, SelectFromModel, SelectKBest,
                                       f_classif, mutual_info_classif)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     TunedThresholdClassifierCV, cross_validate)
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler, OneHotEncoder,
                                   PowerTransformer, RobustScaler, StandardScaler)
from sklearn.svm import LinearSVC

from .baseline import _cat_cols, _cont_cols, cv_schemes
from .data import CAT, CV_SEED
from .features import add_features
from .metrics import PRIMARY, SCORING

# Konfigurasi baseline = persis perlakuan di tahap baseline
BASE_CFG = dict(scaler="standard", encoding="ordinal", resample="none",
                threshold=False, fe=False, fs=None)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Buang salah satu dari pasangan fitur dengan |korelasi| > threshold (dihitung dari data latih)."""

    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        corr = np.nan_to_num(np.abs(np.corrcoef(X, rowvar=False)))
        n, drop = corr.shape[1], set()
        for i in range(n):
            if i in drop:
                continue
            for j in range(i + 1, n):
                if j not in drop and corr[i, j] > self.threshold:
                    drop.add(j)
        self.keep_ = np.array([i for i in range(n) if i not in drop])
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_]


def _scaler(name):
    return {"standard": StandardScaler(), "robust": RobustScaler(), "minmax": MinMaxScaler(),
            "yeojohnson": PowerTransformer(method="yeo-johnson"), "none": "passthrough"}[name]


def _selector(fs, disc_mask):
    if fs is None:
        return None
    name, p = fs
    if name == "anova":
        return SelectKBest(f_classif, k=p)
    if name == "mi":
        return SelectKBest(partial(mutual_info_classif, discrete_features=disc_mask,
                                   random_state=CV_SEED), k=p)
    if name == "rfe":
        return RFE(LinearSVC(dual="auto", max_iter=20000), n_features_to_select=p, step=1)
    if name == "l1":
        return SelectFromModel(LinearSVC(penalty="l1", dual=False, C=p, max_iter=20000),
                               threshold=1e-5)
    if name == "corr":
        return CorrelationFilter(threshold=p)
    raise ValueError(name)


def build_pipeline(base_estimator, X_ref, cfg, memory=None):
    cfg = {**BASE_CFG, **cfg}
    est = clone(base_estimator)
    if cfg["resample"] == "class_weight":
        est.set_params(class_weight="balanced")
    X_fe = add_features(X_ref) if cfg["fe"] else X_ref
    cat = [c for c in CAT if c in X_fe.columns]
    n_cat, n_cont = len(cat), X_fe.shape[1] - len(cat)

    steps = []
    if cfg["fe"]:
        steps.append(("fe", FunctionTransformer(add_features)))
    steps.append(("prep", ColumnTransformer([("num", _scaler(cfg["scaler"]), _cont_cols),
                                             ("cat", "passthrough", _cat_cols)])))
    if cfg["resample"] == "smote":
        steps.append(("smote", SMOTENC(categorical_features=list(range(n_cont, n_cont + n_cat)),
                                       random_state=CV_SEED)))
    disc_mask = np.r_[np.zeros(n_cont, bool), np.ones(n_cat, bool)]
    if cfg["encoding"] == "onehot_nominal" and "Ethnicity" in cat:
        idx = n_cont + cat.index("Ethnicity")
        cats = [float(v) for v in sorted(X_ref["Ethnicity"].unique())]
        steps.append(("ohe", ColumnTransformer(
            [("ohe", OneHotEncoder(categories=[cats], handle_unknown="ignore",
                                   sparse_output=False), [idx])], remainder="passthrough")))
        keep = np.ones(n_cont + n_cat, bool)
        keep[idx] = False
        disc_mask = np.r_[np.ones(len(cats), bool), disc_mask[keep]]
    sel = _selector(cfg["fs"], disc_mask)
    if sel is not None:
        steps.append(("fs", sel))
    steps.append(("clf", est))
    return Pipeline(steps, memory=memory)


def _prefix(grid, p):
    if isinstance(grid, list):
        return [{f"{p}{k}": v for k, v in g.items()} for g in grid]
    return {f"{p}{k}": v for k, v in grid.items()}


def make_search(pipe, grid, cfg, inner, n_jobs=-1):
    """GridSearchCV; jika threshold=True, ambang keputusan ikut dituning (di dalam fold) secara MCC."""
    cfg = {**BASE_CFG, **cfg}
    scorer = SCORING[PRIMARY]
    if cfg["threshold"]:
        tuned = TunedThresholdClassifierCV(
            pipe, scoring=scorer, refit=True,
            cv=StratifiedKFold(5, shuffle=True, random_state=CV_SEED))
        return GridSearchCV(tuned, _prefix(grid, "estimator__"), scoring=scorer,
                            cv=inner, n_jobs=n_jobs, refit=True)
    return GridSearchCV(pipe, grid, scoring=scorer, cv=inner, n_jobs=n_jobs, refit=True)


def fitted_pipeline(search):
    best = search.best_estimator_
    return best.estimator_ if hasattr(best, "estimator_") else best


def hit_max_iter(clf):
    """True jika solver berhenti karena kehabisan iterasi (tidak konvergen). None jika tidak berlaku."""
    mi, ni = getattr(clf, "max_iter", None), getattr(clf, "n_iter_", None)
    if mi is None or ni is None or np.ndim(ni) != 0 or mi <= 0:
        return None
    return bool(ni >= mi)


def evaluate_config(spec, X, y, cfg, grid=None, quick=False, memory=None):
    """Nested CV satu konfigurasi. Protokol & seed SAMA dengan baseline -> fold sama, bisa dipasangkan."""
    inner, outer = cv_schemes(quick)
    pipe = build_pipeline(spec["estimator"], X, cfg, memory=memory)
    search = make_search(pipe, grid if grid is not None else spec["grid"], cfg, inner)
    res = cross_validate(search, X, y, cv=outer, scoring=SCORING, n_jobs=1,
                         return_estimator=True, error_score=np.nan)
    fold_scores = {m: np.asarray(res[f"test_{m}"], dtype=float) for m in SCORING}
    best_params, n_feat, thr, nonconv = [], [], [], 0
    for est in res["estimator"]:
        if not hasattr(est, "best_params_"):
            continue
        best_params.append(est.best_params_)
        clf = fitted_pipeline(est).named_steps["clf"]
        n_feat.append(getattr(clf, "n_features_in_", np.nan))
        nonconv += int(bool(hit_max_iter(clf)))
        if hasattr(est.best_estimator_, "best_threshold_"):
            thr.append(float(est.best_estimator_.best_threshold_))
    extra = {"n_features_final": float(np.nanmean(n_feat)) if n_feat else None,
             "threshold_mean": float(np.mean(thr)) if thr else None,
             "n_nonconverged_folds": float(nonconv)}
    return fold_scores, best_params, extra


def screening_configs(quick=False):
    """Daftar (kelompok, level, cfg). Tiap kelompok diuji TERPISAH terhadap baseline (satu faktor sekali)."""
    ks = [10, 20] if quick else [10, 20, 30, 40]
    out = []
    for s in ["robust", "minmax", "yeojohnson", "none"]:
        out.append(("scaler", s, dict(scaler=s)))
    out.append(("encoding", "onehot_nominal", dict(encoding="onehot_nominal")))
    for rs, th in [("class_weight", False), ("smote", False), ("none", True),
                   ("class_weight", True), ("smote", True)]:
        out.append(("imbalance", f"{rs}{'+threshold' if th else ''}", dict(resample=rs, threshold=th)))
    out.append(("feature_engineering", "fe", dict(fe=True)))
    out.append(("feature_selection", "corr0.9", dict(fs=("corr", 0.9))))
    for k in ks:
        out.append(("feature_selection", f"anova_k{k}", dict(fs=("anova", k))))
    for k in ks:
        out.append(("feature_selection", f"mi_k{k}", dict(fs=("mi", k))))
    for k in ([20] if quick else [20, 30]):
        out.append(("feature_selection", f"rfe_k{k}", dict(fs=("rfe", k))))
    for c in ([0.05] if quick else [0.01, 0.03, 0.1]):
        out.append(("feature_selection", f"l1_C{c}", dict(fs=("l1", c))))
    return out


def final_grid(spec, quick=False):
    """Grid tuning akhir (lebih halus dari baseline).
    Untuk LinearSVC hanya squared_hinge: loss='hinge' memakai solver dual yang pada C besar membutuhkan
    ribuan iterasi dan bisa tidak konvergen (diamati pada data ini). Penalty l1 dan l2 tetap diuji;
    ketidakkonvergenan dipantau lewat n_nonconverged_folds."""
    if spec["svm_type"] == "linearsvc":
        Cs = [float(c) for c in np.logspace(-3, 2, 5 if quick else 11)]
        return {"clf__penalty": ["l2", "l1"], "clf__loss": ["squared_hinge"], "clf__C": Cs}
    return spec["grid"]
