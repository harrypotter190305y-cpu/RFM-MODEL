import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from src.pipeline.exception import CustomException
from src.pipeline.logger import logging
from src.pipeline.utils import save_object, load_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "best_model.pkl")
    pca_model_path: str = os.path.join("artifacts", "pca_model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_rfm, test_rfm, preprocessor_path):
        """
        Train clustering models on RFM data, evaluate them, and perform RFM-based segmentation.
        """
        try:
            logging.info("Starting model training for RFM clustering")

            # --- Load the preprocessor ---
            scaler = load_object(preprocessor_path)
            logging.info("Loaded preprocessor (scaler) for RFM features")

            # --- Prepare RFM Data ---
            # We expect the preprocessor to be fit on raw features: ['Recency','Frequency','Monetary']
            rfm_features = ['Recency', 'Frequency', 'Monetary']
            X_df = train_rfm[rfm_features].copy()

            # remove invalids based on original Frequency/Monetary if present
            if 'Monetary' in train_rfm.columns and 'Frequency' in train_rfm.columns:
                valid_mask = (train_rfm['Monetary'] > 0) & (train_rfm['Frequency'] > 0)
                X_df = X_df[valid_mask.values]

            # --- Scale features ---
            X_scaled = scaler.transform(X_df)

            # --- Automatically determine optimal clusters (2–8) ---
            silhouette_scores = {}
            best_k = 0
            best_score = -1
            
            # Ensure we have enough samples for clustering
            n_samples = len(X_scaled)
            max_clusters = min(8, n_samples - 1)
            min_clusters = min(2, max_clusters)

            for k in range(min_clusters, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                score = silhouette_score(X_scaled, labels)
                silhouette_scores[k] = score
                if score > best_score:
                    best_k, best_score = k, score

            logging.info(f"Optimal number of clusters determined: {best_k}")

            # --- Define models using optimal K (prefer KMeans but evaluate others) ---
            models = {
                "KMeans": KMeans(n_clusters=best_k, random_state=42, n_init=10),
                "Agglomerative": AgglomerativeClustering(n_clusters=best_k),
                "GMM": GaussianMixture(n_components=best_k, random_state=42, reg_covar=1e-6)
            }

            scores = {}

            # --- Train and evaluate each model ---
            for name, model in models.items():
                labels = model.fit_predict(X_scaled)
                silhouette = silhouette_score(X_scaled, labels)
                davies = davies_bouldin_score(X_scaled, labels)
                calinski = calinski_harabasz_score(X_scaled, labels)

                scores[name] = {
                    "Silhouette Score": silhouette,
                    "Davies Bouldin Score": davies,
                    "Calinski Harabasz Score": calinski
                }

                logging.info(f"{name} => Silhouette: {silhouette:.3f}, Davies: {davies:.3f}, Calinski: {calinski:.3f}")

            # --- Select best model based on silhouette score ---
            best_model_name = max(scores, key=lambda x: scores[x]['Silhouette Score'])
            best_model = models[best_model_name]
            # Fit/predict on the subset used for clustering
            try:
                best_labels = best_model.fit_predict(X_scaled)
            except Exception:
                # For GMM use predict after fit
                if hasattr(best_model, 'fit') and hasattr(best_model, 'predict'):
                    best_model.fit(X_scaled)
                    best_labels = best_model.predict(X_scaled)
                else:
                    raise

            logging.info(f"Best clustering model selected: {best_model_name}")

            # --- Apply PCA for Visualization ---
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            # --- Save Best Model and PCA ---
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            save_object(self.model_trainer_config.pca_model_path, pca)

            logging.info(f"Best model saved at: {self.model_trainer_config.trained_model_file_path}")
            logging.info(f"PCA model saved at: {self.model_trainer_config.pca_model_path}")

            # --- Create labeled RFM DataFrame ---
            # Attach cluster labels back to the original train_rfm rows used for clustering
            labeled_df = train_rfm.copy()
            # align labels to the subset used for clustering (X_df)
            labeled_df = labeled_df.loc[X_df.index].copy()
            labeled_df['Cluster'] = best_labels
            labeled_df['PCA1'] = X_pca[:, 0]
            labeled_df['PCA2'] = X_pca[:, 1]

            # --- Create RFM Scores (based on your logic) ---
            # Handle qcut with duplicates by not specifying labels when duplicates='drop'
            # --- Compute R, F, M scores robustly ---
            # Recency: lower is better -> rank descending so most recent customers get highest rank,
            # then qcut into 4 bins where 4=best (most recent).
            labeled_df['R_rank'] = labeled_df['Recency'].rank(method='first', ascending=False)
            try:
                labeled_df['R_Score'] = pd.qcut(labeled_df['R_rank'], 4, labels=[4, 3, 2, 1])
            except ValueError:
                labeled_df['R_Score'] = pd.qcut(labeled_df['R_rank'], 4, duplicates='drop')
                # Map categories to descending scores (4..)
                n_cats = len(labeled_df['R_Score'].cat.categories)
                cats = list(range(4, 4 - n_cats, -1))
                labeled_df['R_Score'] = labeled_df['R_Score'].cat.rename_categories(cats)
            labeled_df['R_Score'] = labeled_df['R_Score'].astype(int)
            labeled_df.drop(columns=['R_rank'], inplace=True)

            # Frequency: higher is better -> rank ascending (smallest rank=lowest freq)
            f_rank = labeled_df['Frequency'].rank(method='first', ascending=True)
            try:
                labeled_df['F_Score'] = pd.qcut(f_rank, 4, labels=[1, 2, 3, 4])
            except ValueError:
                labeled_df['F_Score'] = pd.qcut(f_rank, 4, duplicates='drop')
                n_cats = len(labeled_df['F_Score'].cat.categories)
                cats = list(range(1, 1 + n_cats))
                labeled_df['F_Score'] = labeled_df['F_Score'].cat.rename_categories(cats)
            labeled_df['F_Score'] = labeled_df['F_Score'].astype(int)

            # Monetary: higher is better -> rank ascending then qcut
            m_rank = labeled_df['Monetary'].rank(method='first', ascending=True)
            try:
                labeled_df['M_Score'] = pd.qcut(m_rank, 4, labels=[1, 2, 3, 4])
            except ValueError:
                labeled_df['M_Score'] = pd.qcut(m_rank, 4, duplicates='drop')
                n_cats = len(labeled_df['M_Score'].cat.categories)
                cats = list(range(1, 1 + n_cats))
                labeled_df['M_Score'] = labeled_df['M_Score'].cat.rename_categories(cats)
            labeled_df['M_Score'] = labeled_df['M_Score'].astype(int)

            labeled_df['RFM_Score'] = (
                labeled_df['R_Score'].astype(int) +
                labeled_df['F_Score'].astype(int) +
                labeled_df['M_Score'].astype(int)
            )

            # --- Define segmentation logic ---
            def segment_me(row):
                if row['R_Score'] == 4 and row['F_Score'] == 4 and row['M_Score'] == 4:
                    return 'Champions'
                elif row['F_Score'] >= 3 and row['M_Score'] >= 3 and row['R_Score'] >= 2:
                    return 'Loyal Customers'
                elif row['R_Score'] >= 3 and row['F_Score'] <= 2:
                    return 'Potential Loyalists'
                elif row['R_Score'] >= 3 and row['M_Score'] <= 2:
                    return 'Promising'
                elif row['R_Score'] == 2 and row['F_Score'] <= 2:
                    return 'At Risk'
                elif row['R_Score'] == 1 and row['F_Score'] == 1:
                    return 'Lost'
                else:
                    return 'Others'

            labeled_df['Segment'] = labeled_df.apply(segment_me, axis=1)

            # --- Create Segment Summary ---
            segment_matrix = labeled_df.groupby('Segment').agg({
                'Recency': ['count', 'mean'],
                'Frequency': 'mean',
                'Monetary': 'mean'
            }).reset_index()

            logging.info("Final RFM segmentation matrix created successfully.")
            print("\nFinal RFM Segmentation Matrix:\n", segment_matrix)

            logging.info("Model training and segmentation completed successfully")

            return labeled_df, scores, best_model_name

        except Exception as e:
            raise CustomException(e, sys)