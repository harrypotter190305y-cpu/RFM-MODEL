import os
import sys
import pandas as pd
import zipfile
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.pipeline.exception import CustomException
from src.pipeline.logger import logging


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion paths."""
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):  # ✅ fixed
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Reads customer transaction data, cleans it,
        computes TotalAmount, and splits into train/test datasets.
        """
        logging.info("Entered the data ingestion component for customer segmentation")

        try:
            # ✅ Step 1: Locate dataset
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # ✅ fixed
            # Prefer the full Excel dataset; fall back to other candidates if missing
            possible_paths = [
                os.path.join(base_dir, "data", "Online Retail.xlsx"),
                os.path.join(base_dir, "data", "Online Retail.csv"),
                os.path.join(base_dir, "OnlineRetail.csv"),
                os.path.join(base_dir, "Online Retail.xlsx")
            ]

            data_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    data_path = path
                    break

            if not data_path:
                raise FileNotFoundError("Dataset not found in expected locations.")

            # Log the selected file and its size
            try:
                file_size = os.path.getsize(data_path)
            except Exception:
                file_size = -1
            logging.info(f"Dataset found at: {data_path} (size={file_size} bytes)")

            # ✅ Step 2: Read dataset
            if data_path.lower().endswith(".xlsx"):
                # Attempt to read real Excel file; if it's not a valid xlsx (some files
                # are CSVs renamed to .xlsx) fall back to read_csv.
                try:
                    if not zipfile.is_zipfile(data_path):
                        logging.warning(f"File {data_path} is not a valid xlsx archive — falling back to CSV reader")
                        df = pd.read_csv(data_path)
                    else:
                        df = pd.read_excel(data_path, engine='openpyxl')
                except Exception as ex:
                    logging.warning(f"Failed to read as xlsx ({ex}), trying as CSV")
                    df = pd.read_csv(data_path)
            else:
                df = pd.read_csv(data_path)

            logging.info(f"Dataset loaded successfully with shape: {df.shape}")

            # ✅ Step 3: Basic Cleaning
            df.dropna(subset=['CustomerID'], inplace=True)
            df = df[df['Quantity'] > 0]
            df = df[df['UnitPrice'] > 0]
            df.drop_duplicates(inplace=True)

            logging.info(f"After cleaning: {df.shape[0]} rows remain.")

            # ✅ Step 4: Compute TotalAmount
            if 'TotalAmount' not in df.columns:
                df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

            # ✅ Step 5: Save raw dataset
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Raw dataset saved to {self.ingestion_config.raw_data_path}")

            # ✅ Step 6: Split into train/test
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train-test split completed successfully")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Train and test datasets saved in artifacts folder")

            logging.info("Data ingestion process completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            logging.error("Error occurred during data ingestion")
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    print(f"✅ Data Ingestion Completed.\nTrain Path: {train_data}\nTest Path: {test_data}")
