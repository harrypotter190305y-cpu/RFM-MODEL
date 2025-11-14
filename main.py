import sys
import os

# ✅ Add src folder to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.pipeline.logger import logging

if __name__ == "__main__":
    logging.info("🚀 Starting full RFM pipeline execution...")

    # Step 1️⃣: Data Ingestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    logging.info("✅ Data ingestion completed successfully.")

    # Step 2️⃣: Data Transformation
    transformation = DataTransformation()
    train_rfm, test_rfm, preprocessor_path = transformation.initiate_data_transformation(train_path, test_path)
    logging.info("✅ Data transformation completed successfully.")

    # Step 3️⃣: Model Training
    trainer = ModelTrainer()
    labeled_df, scores, best_model = trainer.initiate_model_trainer(train_rfm, test_rfm, preprocessor_path)
    logging.info("✅ Model training completed successfully.")

    print("\n========= RFM MODEL PIPELINE SUMMARY =========")
    print(f"Best Model Selected : {best_model}")
    print(f"Segmentation Preview:\n{labeled_df.head()}")
    print("==============================================\n")

    logging.info("🏁 Pipeline execution finished successfully.")
