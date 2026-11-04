"""Uji perbandingan berpasangan antar konfigurasi pada fold yang sama."""
import numpy as np
from scipy import stats


def corrected_resampled_ttest(diff, test_train_ratio):
    """Corrected resampled t-test (Nadeau & Bengio, 2003).
    diff: selisih skor per fold (konfigurasi A - B), fold yang sama.
    test_train_ratio: n_test/n_train per fold (5-fold -> 0.25; 3-fold -> 0.5).
    Koreksi diperlukan karena fold CV berulang saling berbagi data latih, sehingga t-test biasa
    terlalu optimistis. Hasilnya tetap indikatif, bukan bukti pasti."""
    d = np.asarray(diff, float)
    d = d[~np.isnan(d)]
    j = len(d)
    out = {"n_folds": j, "mean_diff": float(d.mean()) if j else np.nan, "t": np.nan, "p": np.nan}
    if j < 3 or d.var(ddof=1) == 0:
        return out
    t = d.mean() / np.sqrt((1 / j + test_train_ratio) * d.var(ddof=1))
    out["t"] = float(t)
    out["p"] = float(2 * stats.t.sf(abs(t), j - 1))
    return out
