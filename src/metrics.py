"""
Rancangan metrik evaluasi.

Konteks data: kelas 1 = CKD (mayoritas ~92%), kelas 0 = non-CKD (minoritas ~8%).
Karena kelas positif (CKD) justru mayoritas, metrik yang hanya melihat kelas 1 (recall,
PR-AUC kelas 1) mudah terlihat tinggi. Karena itu SEMUA metrik di bawah dilaporkan untuk
kedua kelas, dan metrik pemilihan model (PRIMARY) harus memakai keempat sel confusion matrix.

PRIMARY  : MCC  (Matthews correlation coefficient)
           - memakai TP, TN, FP, FN sekaligus; tidak berubah jika label kelas ditukar
           - nilai chance/dummy = 0 (mudah ditafsirkan), rentang -1..+1
           Rujukan: Chicco & Jurman (2020), BMC Genomics 21:6.
Sekunder : balanced accuracy, F1-macro, sensitivity (recall CKD), specificity (recall non-CKD),
           ROC-AUC, PR-AUC kelas non-CKD (kelas langka), PR-AUC kelas CKD.
           Rujukan PR-AUC: Saito & Rehmsmeier (2015), PLoS ONE 10:e0118432.
"""
import numpy as np
from sklearn.metrics import (average_precision_score, make_scorer,
                             matthews_corrcoef, recall_score, roc_auc_score)


def _score_ckd(est, X):
    """Skor kontinu: makin tinggi = makin mungkin kelas 1 (CKD)."""
    if hasattr(est, "decision_function"):
        return est.decision_function(X)
    return est.predict_proba(X)[:, 1]


def _roc_auc(est, X, y):
    return roc_auc_score(y, _score_ckd(est, X))


def _pr_auc_ckd(est, X, y):
    return average_precision_score(y, _score_ckd(est, X))


def _pr_auc_nonckd(est, X, y):
    # kelas langka (non-CKD, label 0) dijadikan kelas "positif" untuk PR-AUC
    return average_precision_score((np.asarray(y) == 0).astype(int), -_score_ckd(est, X))


SCORING = {
    "mcc": make_scorer(matthews_corrcoef),
    "balanced_acc": "balanced_accuracy",
    "f1_macro": "f1_macro",
    "sensitivity": make_scorer(recall_score, pos_label=1),   # CKD yang terdeteksi
    "specificity": make_scorer(recall_score, pos_label=0),   # non-CKD yang terdeteksi
    "roc_auc": _roc_auc,
    "pr_auc_nonckd": _pr_auc_nonckd,
    "pr_auc_ckd": _pr_auc_ckd,
}
PRIMARY = "mcc"
