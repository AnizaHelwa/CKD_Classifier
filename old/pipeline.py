"""
CKD - SVM : modul inti (feature engineering, pipeline, grid hyperparameter, evaluasi).
Dipakai bersama oleh semua skrip eksperimen di folder experiments/.
"""
from functools import partial

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import (RFE, SelectFromModel, SelectKBest,
                                       f_classif, mutual_info_classif)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_validate)
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC, NuSVC, LinearSVC

SEED = 42

CAT = [
    "Gender", "Ethnicity", "SocioeconomicStatus", "EducationLevel", "Smoking",
    "FamilyHistoryKidneyDisease", "FamilyHistoryHypertension", "FamilyHistoryDiabetes",
    "PreviousAcuteKidneyInjury", "UrinaryTractInfections", "ACEInhibitors", "Diuretics",
    "Statins", "AntidiabeticMedications", "Edema", "HeavyMetalsExposure",
    "OccupationalExposureChemicals", "WaterQuality",
]

# Metrik evaluasi. f1_macro = metrik UTAMA untuk memilih model (adil ke kelas minoritas).
# Sisanya dilaporkan sebagai metrik sekunder, tidak dipakai untuk memilih model terbaik.
SCORING = {
    "f1_macro": "f1_macro",
    "balanced_acc": "balanced_accuracy",
    "recall_pos": "recall",
    "roc_auc": "roc_auc",
}
PRIMARY_METRIC = "f1_macro"


# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------
def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    c, eps = set(X.columns), 1e-6
    if {"SystolicBP", "DiastolicBP"} <= c:
        X["MAP"] = (X["SystolicBP"] + 2 * X["DiastolicBP"]) / 3
        X["PulsePressure"] = X["SystolicBP"] - X["DiastolicBP"]
    if {"BUNLevels", "SerumCreatinine"} <= c:
        X["BUN_Cr_ratio"] = X["BUNLevels"] / (X["SerumCreatinine"] + eps)
    if {"CholesterolLDL", "CholesterolHDL"} <= c:
        X["LDL_HDL_ratio"] = X["CholesterolLDL"] / (X["CholesterolHDL"] + eps)
    if {"CholesterolTotal", "CholesterolHDL"} <= c:
        X["TC_HDL_ratio"] = X["CholesterolTotal"] / (X["CholesterolHDL"] + eps)
    for col in ["SerumCreatinine", "ProteinInUrine"]:
        if col in c:
            X[f"log_{col}"] = np.log1p(X[col].clip(lower=0))
    sym = [s for s in ["Itching", "FatigueLevels", "NauseaVomiting", "MuscleCramps"] if s in c]
    if sym:
        X["SymptomBurden"] = X[sym].sum(axis=1)
    meds = [m for m in ["ACEInhibitors", "Diuretics", "Statins",
                        "AntidiabeticMedications", "NSAIDsUse"] if m in c]
    if meds:
        X["MedicationCount"] = X[meds].sum(axis=1)
    fam = [f for f in c if f.startswith("FamilyHistory")]
    if fam:
        X["FamilyHistoryCount"] = X[fam].sum(axis=1)
    return X


def _cont_cols(X):
    return [col for col in X.columns if col not in CAT]


# ----------------------------------------------------------------------------
# Seleksi fitur
# ----------------------------------------------------------------------------
def make_selector(fs, n_cont, n_cat):
    if fs is None:
        return None
    name, param = fs
    if name == "anova":
        return SelectKBest(f_classif, k=param)
    if name == "mi":
        mask = np.r_[np.zeros(n_cont, bool), np.ones(n_cat, bool)]
        return SelectKBest(partial(mutual_info_classif, discrete_features=mask,
                                   random_state=SEED), k=param)
    if name == "rfe":
        return RFE(LogisticRegression(max_iter=2000, solver="liblinear", random_state=SEED),
                   n_features_to_select=param)
    if name == "l1":
        return SelectFromModel(
            LogisticRegression(penalty="l1", C=param, solver="liblinear",
                               max_iter=2000, random_state=SEED), threshold=1e-5)
    raise ValueError(name)


# ----------------------------------------------------------------------------
# Varian SVM
# ----------------------------------------------------------------------------
def make_svm(svm_type, kernel, class_weight=None):
    """svm_type: 'svc' | 'nusvc' | 'linearsvc' (linearsvc hanya untuk kernel linear)."""
    if svm_type == "svc":
        return SVC(kernel=kernel, class_weight=class_weight)
    if svm_type == "nusvc":
        return NuSVC(kernel=kernel, class_weight=class_weight, nu=0.5)
    if svm_type == "linearsvc":
        if kernel != "linear":
            raise ValueError("LinearSVC hanya mendukung kernel linear")
        return LinearSVC(class_weight=class_weight, max_iter=5000, dual="auto")
    raise ValueError(svm_type)


def svm_step_name(svm_type):
    return "svm"


# ----------------------------------------------------------------------------
# Pipeline builder
# ----------------------------------------------------------------------------
def build_pipeline(X_ref, svm_type="svc", kernel="rbf", fs=None, fe=False,
                   resample="smote", memory=None):
    X_fe = add_features(X_ref) if fe else X_ref
    cat = [c for c in CAT if c in X_fe.columns]
    n_cat = len(cat)
    n_cont = X_fe.shape[1] - n_cat

    steps = []
    if fe:
        steps.append(("fe", FunctionTransformer(add_features)))
    steps.append(("prep", ColumnTransformer(
        [("num", StandardScaler(), _cont_cols), ("cat", "passthrough", cat)])))
    if resample == "smote":
        steps.append(("smote", SMOTENC(categorical_features=list(range(n_cont, n_cont + n_cat)),
                                       random_state=SEED)))
    sel = make_selector(fs, n_cont, n_cat)
    if sel is not None:
        steps.append(("fs", sel))
    class_weight = "balanced" if resample == "class_weight" else None
    steps.append(("svm", make_svm(svm_type, kernel, class_weight)))
    return Pipeline(steps, memory=memory)


# ----------------------------------------------------------------------------
# Grid hyperparameter per kombinasi (svm_type, kernel)
# ----------------------------------------------------------------------------
def kernel_grid(svm_type, kernel):
    Cs = np.logspace(-2, 3, 6)
    nus = [0.1, 0.3, 0.5, 0.7, 0.9]
    gam = np.logspace(-4, 0, 5)

    param_prefix = "svm__"

    if svm_type == "linearsvc":
        return {f"{param_prefix}C": Cs}

    C_or_nu = {f"{param_prefix}nu": nus} if svm_type == "nusvc" else {f"{param_prefix}C": Cs}

    if kernel == "linear":
        return dict(C_or_nu)
    if kernel == "rbf":
        return {**C_or_nu, f"{param_prefix}gamma": ["scale", *gam]}
    if kernel == "poly":
        return {**C_or_nu, f"{param_prefix}degree": [2, 3],
                f"{param_prefix}gamma": ["scale"], f"{param_prefix}coef0": [0, 1]}
    if kernel == "sigmoid":
        return {**C_or_nu, f"{param_prefix}gamma": ["scale", *gam[:3]],
                f"{param_prefix}coef0": [0, 1]}
    raise ValueError(kernel)


# ----------------------------------------------------------------------------
# Cross-validation objects dan evaluasi nested
# ----------------------------------------------------------------------------
def cv_objects(quick=False):
    if quick:
        return (StratifiedKFold(3, shuffle=True, random_state=SEED),
                RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=SEED))
    return (StratifiedKFold(3, shuffle=True, random_state=SEED),
            RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED))


def nested_eval(label, pipe, grid, X, y, quick=False):
    inner, outer = cv_objects(quick)
    gs = GridSearchCV(pipe, grid, scoring=PRIMARY_METRIC, cv=inner, n_jobs=-1)
    res = cross_validate(gs, X, y, cv=outer, scoring=SCORING, n_jobs=1,
                         return_estimator=True)
    row = {"config": label}
    for m in SCORING:
        s = res[f"test_{m}"]
        row[m] = f"{s.mean():.3f} ± {s.std():.3f}"
        row[f"_{m}_mean"] = s.mean()
        row[f"_{m}_std"] = s.std()
    # Hyperparameter terbaik yang paling sering muncul di antar fold (modus)
    best_params_list = [est.best_params_ for est in res["estimator"]]
    row["_best_params_per_fold"] = best_params_list
    return row


def l1_sweep(X, y, svm_type, kernel, Cs, fe=False, resample="smote", quick=False):
    """Sweep nilai C (alpha=1/C) untuk seleksi fitur L1. Mengembalikan list of dict,
    satu per nilai C, siap di-log satu-per-satu ke MLflow."""
    _, outer = cv_objects(quick)
    rows = []
    for C in Cs:
        pipe = build_pipeline(X, svm_type=svm_type, kernel=kernel, fs=("l1", C),
                              fe=fe, resample=resample)
        try:
            res = cross_validate(pipe, X, y, cv=outer, scoring=PRIMARY_METRIC,
                                 return_estimator=True, n_jobs=-1, error_score=np.nan)
        except ValueError:
            rows.append({"C": C, "alpha": 1 / C, "n_features": 0,
                         "f1_mean": np.nan, "f1_se": np.nan})
            continue
        nfeat = [est.named_steps["fs"].get_support().sum()
                 for est in res["estimator"] if hasattr(est.named_steps["fs"], "estimator_")]
        s = res["test_score"]
        s = s[~np.isnan(s)]
        rows.append({"C": C, "alpha": 1 / C,
                     "n_features": float(np.mean(nfeat)) if nfeat else 0,
                     "f1_mean": float(s.mean()) if len(s) else np.nan,
                     "f1_se": float(s.std() / np.sqrt(len(s))) if len(s) else np.nan})
    return rows
