import os
import sys
from dataclasses import dataclass
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from src.pipeline.exception import CustomException
from src.pipeline.logger import logging
from src.pipeline.utils import save_object
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class DataTransformationConfig:
    """Configuration for saving the preprocessor object."""
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):  # ✅ Fixed: double underscores
        self.data_transformation_config = DataTransformationConfig()

    def create_rfm_features(self, df):
        """
        Create RFM (Recency, Frequency, Monetary) features from transaction data.
        Includes data cleaning and consistency with notebook prototype.
        """
        try:
            logging.info("Starting RFM feature creation...")

            # --- Data Cleaning ---
            df = df.dropna(subset=['CustomerID'])
            df = df[df['Quantity'] > 0]
            df = df.drop_duplicates()

            # Compute total amount if missing
            if "TotalAmount" not in df.columns:
                df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

            # Convert InvoiceDate to datetime
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

            # Reference date (latest invoice + 1 day)
            ref_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

            # --- Calculate RFM ---
            rfm = df.groupby('CustomerID').agg({
                'InvoiceDate': lambda x: (ref_date - x.max()).days,  # Recency
                'InvoiceNo': 'nunique',                              # Frequency
                'TotalAmount': 'sum'                                 # Monetary
            }).reset_index()

            rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

            logging.info(f"✅ RFM features created successfully. Shape: {rfm.shape}")
            return rfm

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Calculates RFM features from transaction data and applies StandardScaler.
        Returns RFM dataframes and saves the scaler as preprocessor.
        """
        try:
            logging.info("📥 Loading train and test data...")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("✅ Train and test data loaded successfully.")

            # --- Create RFM features ---
            train_rfm = self.create_rfm_features(train_df)
            test_rfm = self.create_rfm_features(test_df)

            # --- Feature selection ---
            # Use raw RFM features (no log transform) and fit a RobustScaler.
            # The experiment grid showed raw + RobustScaler works well for this dataset.
            scaler = RobustScaler()
            rfm_features = ['Recency', 'Frequency', 'Monetary']

            # Fit scaler on training RFM features
            scaler.fit(train_rfm[rfm_features])

            # Save the preprocessor (scaler)
            save_object(
                self.data_transformation_config.preprocessor_obj_file_path,
                scaler
            )

            logging.info(f"✅ Scaler saved at {self.data_transformation_config.preprocessor_obj_file_path}")

            return (
                train_rfm,
                test_rfm,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    logging.info("Data Transformation module executed successfully")
