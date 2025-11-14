import sys
import os
import pandas as pd
import numpy as np
from src.pipeline.exception import CustomException
from src.pipeline.logger import logging
from src.pipeline.utils import load_object
from sklearn.preprocessing import StandardScaler

class PredictPipeline:
    def __init__(self):
        # Paths to saved models (same as in model_trainer.py)
        self.model_path = os.path.join("artifacts", "best_model.pkl")
        self.pca_path = os.path.join("artifacts", "pca_model.pkl")

    def predict(self, features):
        try:
            logging.info("Loading saved clustering model and PCA object...")
            model = load_object(file_path=self.model_path)
            pca = load_object(file_path=self.pca_path)

            # Convert input features to DataFrame if not already
            if isinstance(features, dict):
                data = pd.DataFrame([features])
            elif isinstance(features, pd.DataFrame):
                data = features
            else:
                raise CustomException("Invalid input type. Expected dict or DataFrame.")

            # Extract only required columns
            X = data[['Recency', 'Frequency', 'Monetary']]

            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Predict cluster
            if hasattr(model, "predict"):
                cluster_labels = model.predict(X_scaled)
            else:
                # AgglomerativeClustering has no .predict(), so use fit_predict
                cluster_labels = model.fit_predict(X_scaled)

            # Apply PCA for visualization or dimensionality reduction
            X_pca = pca.transform(X_scaled)

            # Combine results into a clean DataFrame
            result_df = pd.DataFrame({
                'Recency': X['Recency'],
                'Frequency': X['Frequency'],
                'Monetary': X['Monetary'],
                'Cluster': cluster_labels,
                'PCA1': X_pca[:, 0],
                'PCA2': X_pca[:, 1]
            })

            logging.info("Prediction completed successfully.")
            return result_df

        except Exception as e:
            raise CustomException(e, sys)
