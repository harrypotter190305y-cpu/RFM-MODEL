import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from src.pipeline.exception import CustomException
from src.pipeline.logger import logging

###############################################################
# ✅ Save any object (model, transformer, PCA, scaler, etc.)
###############################################################
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

###############################################################
# Backwards/forwards-compatible save_object wrapper
###############################################################
def save_object_compatible(*args, **kwargs):
    """Flexible saver that accepts multiple calling patterns:

    Patterns supported:
      save_object_compatible(obj, path=...) or save_object_compatible(obj, file_path=...)
      save_object_compatible(path, obj)  # positional (file_path, obj)
      save_object_compatible(obj, path)  # positional (obj, path)
      save_object_compatible(file_path=..., obj=...)
    """
    try:
        # Extract from kwargs first
        file_path = kwargs.get("file_path") or kwargs.get("path")
        obj = kwargs.get("obj")

        # Handle positional args
        if len(args) == 2:
            a0, a1 = args[0], args[1]
            # heuristic: strings are paths
            if isinstance(a0, str) and not isinstance(a1, str):
                file_path, obj = a0, a1
            elif isinstance(a1, str) and not isinstance(a0, str):
                file_path, obj = a1, a0
            else:
                # fallback: assume (obj, path)
                obj, file_path = a0, a1
        elif len(args) == 1:
            # single positional arg: ambiguous, assume it's the object if not a str
            if isinstance(args[0], str):
                file_path = file_path or args[0]
            else:
                obj = obj or args[0]

        if file_path is None or obj is None:
            raise ValueError("save_object requires both an object and a file path (obj & path/file_path)")

        dir_path = os.path.dirname(file_path) or "."
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

# Export a stable name `save_object` to be tolerant of callers
save_object = save_object_compatible

###############################################################
# ✅ Load any saved object
###############################################################
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

###############################################################
# ✅ Evaluate clustering models using multiple metrics
###############################################################
def evaluate_clustering_models(X_scaled, models):
    """
    Evaluate different clustering models using multiple metrics.
    Returns a dictionary with silhouette, Davies-Bouldin, and Calinski-Harabasz scores.
    """
    try:
        report = {}

        for name, model in models.items():
            logging.info(f"Training and evaluating {name} model...")

            # Some models don't have predict (e.g., Agglomerative)
            if hasattr(model, "fit_predict"):
                labels = model.fit_predict(X_scaled)
            else:
                model.fit(X_scaled)
                labels = model.predict(X_scaled)

            # Compute clustering metrics
            sil_score = silhouette_score(X_scaled, labels)
            db_score = davies_bouldin_score(X_scaled, labels)
            ch_score = calinski_harabasz_score(X_scaled, labels)

            report[name] = {
                "Silhouette": sil_score,
                "Davies-Bouldin": db_score,
                "Calinski-Harabasz": ch_score
            }

            logging.info(f"{name} → Silhouette: {sil_score:.4f}, DB: {db_score:.4f}, CH: {ch_score:.4f}")

        return report

    except Exception as e:
        raise CustomException(e, sys)

