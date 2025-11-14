import os
import sys
from src.pipeline.exception import CustomException
from src.pipeline.logger import logging

# Import all major pipeline components
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    try:
        logging.info("====== Customer Segmentation Training Pipeline Started ======")

        ############################################################
        # 1️⃣ DATA INGESTION
        ############################################################
        ingestion_obj = DataIngestion()
        train_data_path, test_data_path = ingestion_obj.initiate_data_ingestion()
        logging.info(f"✅ Data Ingestion Completed. Train: {train_data_path}, Test: {test_data_path}")

        ############################################################
        # 2️⃣ DATA TRANSFORMATION (RFM Feature Engineering)
        ############################################################
        transformation_obj = DataTransformation()
        train_rfm, test_rfm, preprocessor_path = transformation_obj.initiate_data_transformation(
            train_path=train_data_path,
            test_path=test_data_path
        )
        logging.info(f"✅ Data Transformation Completed. RFM features created and preprocessor saved at: {preprocessor_path}")

        ############################################################
        # 3️⃣ MODEL TRAINING (Clustering on RFM Data)
        ############################################################
        model_trainer_obj = ModelTrainer()
        labeled_rfm, scores, best_model_name = model_trainer_obj.initiate_model_trainer(
            train_rfm=train_rfm,
            test_rfm=test_rfm,
            preprocessor_path=preprocessor_path
        )

        logging.info(f"✅ Model Training Completed. Best Model: {best_model_name}")
        logging.info(f"Model Scores: {scores[best_model_name]}")

        ############################################################
        # ✅ PIPELINE COMPLETION
        ############################################################
        logging.info("====== Training Pipeline Summary ======")
        logging.info(f"Total customers clustered: {len(labeled_rfm)}")
        logging.info(f"Cluster distribution:\n{labeled_rfm['Cluster'].value_counts().sort_index()}")
        logging.info("")
        logging.info("Files generated in artifacts folder:")
        logging.info("  1. raw_data.csv - Raw transaction data")
        logging.info("  2. train.csv - Training transaction data")
        logging.info("  3. test.csv - Test transaction data")
        logging.info("  4. preprocessor.pkl - StandardScaler for RFM features")
        logging.info("  5. best_model.pkl - Best clustering model")
        logging.info("  6. pca_model.pkl - PCA model for visualization")
        logging.info("")
        logging.info("🎯 Customer Segmentation Training Pipeline executed successfully!")

    except Exception as e:
        logging.error("❌ Error occurred during the training pipeline execution")
        raise CustomException(e, sys)

