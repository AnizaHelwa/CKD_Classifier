"""
Pemuatan data + pembagian train/test.

ATURAN: test set TIDAK BOLEH disentuh pada tahap baseline maupun ablasi. Fungsi
load_train() sengaja hanya mengembalikan data latih. Test set baru dibuka sekali
di tahap evaluasi final lewat load_split().
"""
import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split

CV_SEED = 42          # seed untuk semua cross-validation (sama untuk semua kandidat -> perbandingan berpasangan)
SPLIT_SEED = 2026     # seed khusus pembagian train/test (split baru, terpisah dari notebook lama)
TEST_SIZE = 0.2
TARGET = "Diagnosis"  # 1 = CKD (kelas mayoritas, ~92%), 0 = non-CKD (kelas minoritas, ~8%)
DROP_COLS = ["PatientID", "DoctorInCharge"]

CAT = [
    "Gender", "Ethnicity", "SocioeconomicStatus", "EducationLevel", "Smoking",
    "FamilyHistoryKidneyDisease", "FamilyHistoryHypertension", "FamilyHistoryDiabetes",
    "PreviousAcuteKidneyInjury", "UrinaryTractInfections", "ACEInhibitors", "Diuretics",
    "Statins", "AntidiabeticMedications", "Edema", "HeavyMetalsExposure",
    "OccupationalExposureChemicals", "WaterQuality",
]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_split(path="data/CKD.csv"):
    """Mengembalikan X_train, X_test, y_train, y_test. Hanya dipakai di evaluasi final."""
    df = pd.read_csv(path, sep=";").drop(columns=DROP_COLS, errors="ignore")
    X, y = df.drop(columns=TARGET), df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y)


def load_train(path="data/CKD.csv"):
    """Hanya data latih. Dipakai baseline dan ablasi."""
    X_train, _, y_train, _ = load_split(path)
    return X_train, y_train
