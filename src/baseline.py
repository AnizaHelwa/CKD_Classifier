"""
Definisi kandidat baseline + protokol evaluasi (nested CV).

Prinsip baseline (SAMA untuk semua kandidat, supaya perbandingan adil):
  * preprocessing minimal: StandardScaler untuk fitur kontinu, fitur kategorikal dibiarkan
  * TANPA resampling (SMOTE), TANPA class_weight, TANPA seleksi fitur, TANPA feature engineering
    -> semua itu adalah komponen tahap berikutnya (ablasi), bukan bagian baseline
  * tuning hyperparameter tiap kandidat: GridSearchCV 3-fold, dipilih berdasar metrik PRIMARY
  * evaluasi: RepeatedStratifiedKFold 5x3 dengan seed yang sama untuk semua kandidat
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC, NuSVC

from .data import CAT, CV_SEED
from .metrics import PRIMARY, SCORING

KERNELS = ["linear", "poly", "rbf", "sigmoid"]


def _cont_cols(X):
    return [c for c in X.columns if c not in CAT]


def _cat_cols(X):
    return [c for c in CAT if c in X.columns]


def _make_pipeline(estimator):
    prep = ColumnTransformer([("num", StandardScaler(), _cont_cols),
                              ("cat", "passthrough", _cat_cols)])
    return Pipeline([("prep", prep), ("clf", estimator)])


def nu_grid(y, safety=0.85):
    """Grid nu untuk NuSVC yang dijamin feasible.

    nu harus <= 2*min(proporsi kelas) (Crisp & Burges, 2000; Chang & Lin, 2001), kalau tidak
    libsvm melempar 'specified nu is infeasible'. safety<1 memberi ruang untuk variasi proporsi
    kelas di dalam fold. Grid = pecahan dari batas atas tersebut."""
    p = float(min(np.mean(np.asarray(y) == 0), np.mean(np.asarray(y) == 1)))
    nu_max = 2 * p * safety
    return [round(nu_max * f, 4) for f in (0.15, 0.3, 0.5, 0.75, 1.0)]


def _svm_grid(kernel, reg_name, reg_values):
    gam = [float(g) for g in np.logspace(-4, 0, 5)]
    g = {f"clf__{reg_name}": reg_values}
    if kernel == "rbf":
        g["clf__gamma"] = ["scale", *gam]
    elif kernel == "poly":
        g.update({"clf__degree": [2, 3], "clf__gamma": ["scale"], "clf__coef0": [0, 1]})
    elif kernel == "sigmoid":
        g.update({"clf__gamma": ["scale", *gam[:3]], "clf__coef0": [0, 1]})
    return g


def build_candidates(y_train, quick=False):
    """dict nama -> spec. family: 'SVM' (boleh jadi pemenang), 'comparator'/'reference' (konteks)."""
    Cs = [float(c) for c in np.logspace(-2, 3, 3 if quick else 6)]
    nus = nu_grid(y_train)
    if quick:
        nus = nus[::2]

    cands = {}
    cands["dummy_most_frequent"] = dict(
        family="reference", svm_type="-", kernel="-", grid={},
        estimator=DummyClassifier(strategy="most_frequent"))
    cands["logistic_regression"] = dict(
        family="comparator", svm_type="-", kernel="-",
        grid={"clf__C": [float(c) for c in np.logspace(-2, 2, 3 if quick else 5)]},
        estimator=LogisticRegression(max_iter=5000, random_state=CV_SEED))
    cands["random_forest"] = dict(
        family="comparator", svm_type="-", kernel="-",
        grid={"clf__n_estimators": [100] if quick else [100, 300],
              "clf__max_depth": [None, 10] if quick else [None, 10, 20],
              "clf__min_samples_leaf": [1] if quick else [1, 3]},
        estimator=RandomForestClassifier(random_state=CV_SEED, n_jobs=1))
    cands["linearsvc_linear"] = dict(
        family="SVM", svm_type="linearsvc", kernel="linear",
        grid={"clf__C": Cs}, estimator=LinearSVC(dual="auto", max_iter=20000))
    for k in KERNELS:
        cands[f"svc_{k}"] = dict(family="SVM", svm_type="svc", kernel=k,
                                 grid=_svm_grid(k, "C", Cs), estimator=SVC(kernel=k))
    for k in KERNELS:
        cands[f"nusvc_{k}"] = dict(family="SVM", svm_type="nusvc", kernel=k,
                                   grid=_svm_grid(k, "nu", nus), estimator=NuSVC(kernel=k))
    return cands


def cv_schemes(quick=False):
    inner = StratifiedKFold(3, shuffle=True, random_state=CV_SEED)
    if quick:
        outer = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=CV_SEED)
    else:
        outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=CV_SEED)
    return inner, outer


def evaluate_candidate(spec, X, y, quick=False):
    """Nested CV untuk satu kandidat. Mengembalikan (fold_scores: dict metrik->array, best_params: list)."""
    inner, outer = cv_schemes(quick)
    pipe = _make_pipeline(spec["estimator"])
    if spec["grid"]:
        model = GridSearchCV(pipe, spec["grid"], scoring=SCORING[PRIMARY],
                             cv=inner, n_jobs=-1, refit=True)
    else:
        model = pipe
    res = cross_validate(model, X, y, cv=outer, scoring=SCORING, n_jobs=1,
                         return_estimator=True, error_score=np.nan)
    fold_scores = {m: np.asarray(res[f"test_{m}"], dtype=float) for m in SCORING}
    best_params = [getattr(est, "best_params_", {}) for est in res["estimator"]]
    return fold_scores, best_params
