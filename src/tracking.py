"""Pencatatan ke MLflow yang dipakai bersama oleh semua skrip tahap 2."""
import json
import os
import tempfile

import mlflow
import numpy as np
import pandas as pd

from .metrics import PRIMARY


def already_done(experiment, name, mode, stage):
    runs = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string=f"tags.mode = '{mode}' and tags.stage = '{stage}' "
                      f"and tags.mlflow.runName = '{name}' and attributes.status = 'FINISHED'")
    return len(runs) > 0


def find_run(experiment, name, mode, stage):
    runs = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string=f"tags.mode = '{mode}' and tags.stage = '{stage}' "
                      f"and tags.mlflow.runName = '{name}' and attributes.status = 'FINISHED'")
    return None if runs.empty else runs.iloc[0]


def log_run(name, tags, params, fold_scores, best_params, extra=None):
    with mlflow.start_run(run_name=name):
        mlflow.set_tags(tags)
        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        mlflow.log_metric("n_failed_folds", int(np.isnan(fold_scores[PRIMARY]).sum()))
        for m, s in fold_scores.items():
            n = max(int(np.sum(~np.isnan(s))), 1)
            mlflow.log_metric(f"{m}_mean", float(np.nanmean(s)))
            mlflow.log_metric(f"{m}_std", float(np.nanstd(s)))
            mlflow.log_metric(f"{m}_se", float(np.nanstd(s) / np.sqrt(n)))
        for k, v in (extra or {}).items():
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                mlflow.log_metric(k, float(v))
        with tempfile.TemporaryDirectory() as d:
            pd.DataFrame(fold_scores).to_csv(os.path.join(d, "fold_scores.csv"), index_label="fold")
            with open(os.path.join(d, "best_params_per_fold.json"), "w", encoding="utf-8") as f:
                json.dump(best_params, f, default=str)
            mlflow.log_artifacts(d)


def load_fold_scores(run_id):
    p = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="fold_scores.csv")
    return pd.read_csv(p, index_col="fold")
