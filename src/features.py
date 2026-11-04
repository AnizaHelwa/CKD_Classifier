"""Feature engineering berbasis domain klinis. Baris-wise, tanpa fitting -> aman dari data leakage."""
import numpy as np
import pandas as pd

EPS = 1e-6


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    c = set(X.columns)
    if {"SystolicBP", "DiastolicBP"} <= c:
        X["MAP"] = (X["SystolicBP"] + 2 * X["DiastolicBP"]) / 3
        X["PulsePressure"] = X["SystolicBP"] - X["DiastolicBP"]
    if {"BUNLevels", "SerumCreatinine"} <= c:
        X["BUN_Cr_ratio"] = X["BUNLevels"] / (X["SerumCreatinine"] + EPS)
    if {"CholesterolLDL", "CholesterolHDL"} <= c:
        X["LDL_HDL_ratio"] = X["CholesterolLDL"] / (X["CholesterolHDL"] + EPS)
    if {"CholesterolTotal", "CholesterolHDL"} <= c:
        X["TC_HDL_ratio"] = X["CholesterolTotal"] / (X["CholesterolHDL"] + EPS)
    for col in ["SerumCreatinine", "ProteinInUrine"]:          # distribusi miring kanan
        if col in c:
            X[f"log_{col}"] = np.log1p(X[col].clip(lower=0))
    sym = [s for s in ["Itching", "FatigueLevels", "NauseaVomiting", "MuscleCramps"] if s in c]
    if sym:
        X["SymptomBurden"] = X[sym].sum(axis=1)
    # hanya obat biner; NSAIDsUse sengaja tidak dijumlahkan (skalanya bukan 0/1)
    meds = [m for m in ["ACEInhibitors", "Diuretics", "Statins", "AntidiabeticMedications"] if m in c]
    if meds:
        X["MedicationCount"] = X[meds].sum(axis=1)
    fam = [f for f in c if f.startswith("FamilyHistory")]
    if fam:
        X["FamilyHistoryCount"] = X[fam].sum(axis=1)
    return X


def created_features(X: pd.DataFrame):
    """Fitur yang BENAR-BENAR terbentuk. Cetak ini saat eksperimen supaya nama kolom yang
    salah ketik/tidak ada tidak diam-diam menonaktifkan fitur."""
    base = set(X.columns)
    return [c for c in add_features(X.head(20)).columns if c not in base]
