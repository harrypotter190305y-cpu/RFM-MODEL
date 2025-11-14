# RFM Customer Segmentation Model

## Overview
This project implements an **RFM (Recency, Frequency, Monetary) customer segmentation analysis** using machine learning. The system analyzes customer purchase behavior and automatically assigns customers to 7 distinct segments for targeted marketing strategies.

## Project Structure
```
RFM MODEL/
├── artifacts/              # Saved models, scalers, and results
│   ├── best_model.pkl      # Trained clustering model
│   ├── preprocessor.pkl    # Fitted scaler (RobustScaler)
│   ├── pca_model.pkl       # PCA for visualization
│   ├── raw_data.csv        # Cleaned transaction data
│   ├── train.csv           # Training dataset
│   └── test.csv            # Test dataset
├── data/                   # Raw input data
│   └── Online Retail.xlsx  # Online Retail transaction dataset (~23MB)
├── logs/                   # Pipeline execution logs
│   └── [timestamp].log     # Timestamped log files
├── src/
│   ├── components/         # Core processing modules
│   │   ├── data_ingestion.py          # Load and clean data
│   │   ├── data_transformation.py     # Create RFM features and scaling
│   │   └── model_trainer.py           # Train clustering and segment
│   └── pipeline/           # Utilities and configuration
│       ├── logger.py       # Logging setup
│       ├── exception.py    # Error handling
│       ├── utils.py        # Helper functions
│       ├── train_pipeline.py
│       └── predict_pipeline.py
├── templates/              # HTML templates for Flask UI
│   ├── home.html
│   ├── dashboard.html      # Main segmentation dashboard
│   └── dataset.html        # Dataset preview
├── main.py                 # Pipeline entry point (CLI)
├── app.py                  # Flask web app entry point
└── requirements.txt        # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup
1. Clone or navigate to the project directory:
   ```bash
   cd "C:\RFM MODEL\RFM MODEL"
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the Pipeline (CLI)
Generate RFM features and train the segmentation model:
```bash
python main.py
```

This will:
- Load the Online Retail dataset
- Create RFM features for each customer
- Train a clustering model
- Generate 7 customer segments
- Save artifacts (model, scaler, predictions) to `artifacts/`
- Print a segmentation summary

### Run the Web Dashboard
Start the interactive Flask web application:
```bash
python app.py
```
or
```bash
flask run
```

Then open http://127.0.0.1:5000 in your browser to:
- View the home page
- Explore the dataset preview
- Generate and visualize customer segments
- Download segment statistics

## RFM Analysis

### What is RFM?
RFM segmentation is a data-driven approach to divide customers into groups based on:

- **Recency (R)**: Days since the customer's last purchase
  - **Lower is better** (more recent purchases = higher value)
  
- **Frequency (F)**: Number of times the customer has purchased
  - **Higher is better** (more frequent purchases = higher loyalty)
  
- **Monetary (M)**: Total amount the customer has spent
  - **Higher is better** (higher spending = higher value)

Each metric is scored on a scale of 1-4, and combined to create an RFM score (3-12).

### The 7 Segments

| Segment | Characteristics | Business Strategy |
|---------|-----------------|------------------|
| **Champions** | Highest R, F, M scores; recent, frequent, big spenders | Reward loyalty; introduce new products |
| **Loyal Customers** | High F and M; good retention | Increase frequency; cross-sell/upsell |
| **Potential Loyalists** | Moderate frequency, recent; lower spend | Nurture with personalized offers |
| **Promising** | Recent but infrequent; lower spend | Build engagement; recommend products |
| **At Risk** | Low recency; infrequent; lower spend | Win-back campaigns; special offers |
| **Lost** | Very low recency and frequency | Re-engagement campaigns or de-list |
| **Others** | Unique behavior patterns | Segment-specific strategies |

## Data Processing

### Data Ingestion (`data_ingestion.py`)
- Loads Online Retail dataset from `data/Online Retail.xlsx`
- Cleans data: removes missing CustomerID, invalid quantities, duplicates
- Splits into train/test sets (80/20)
- Saves to `artifacts/raw_data.csv`, `artifacts/train.csv`, `artifacts/test.csv`

### Data Transformation (`data_transformation.py`)
- Creates RFM metrics for each customer:
  - Recency: days since last purchase
  - Frequency: count of unique invoices
  - Monetary: sum of purchase amounts
- Fits `RobustScaler` on training RFM features to handle outliers
- Saves preprocessor to `artifacts/preprocessor.pkl`

### Model Training (`model_trainer.py`)
- Determines optimal number of clusters (k=2-8) using silhouette score
- Trains three clustering models: KMeans, Agglomerative, GaussianMixture
- Selects best model (typically KMeans or Agglomerative) based on silhouette score
- Applies RFM scoring (1-4 per metric) using quantile-based ranking
- Maps RFM scores to 7 business segments using predefined rules
- Saves model and artifacts to `artifacts/`

## Technologies Used
- **Python 3.x**
- **Pandas** - Data processing and RFM feature engineering
- **NumPy** - Numerical operations
- **Scikit-learn** - Clustering (KMeans, Agglomerative, GaussianMixture) and scaling (RobustScaler)
- **Flask** - Web framework for interactive dashboard
- **Matplotlib & Seaborn** - Visualization
- **Dill** - Model serialization

## Key Features
✅ **Automated RFM Calculation** - Computes metrics for all customers  
✅ **Multiple Clustering Algorithms** - KMeans, Agglomerative, GMM with automatic model selection  
✅ **7 Interpretable Segments** - Business-friendly segment names and descriptions  
✅ **Web Dashboard** - Interactive UI to generate and visualize segments  
✅ **Data Caching** - Efficient preprocessing and model loading  
✅ **Comprehensive Logging** - Timestamped logs for debugging  
✅ **Production-Ready** - Scalable pipeline with error handling  

## Performance Metrics
The pipeline evaluates clustering quality using:
- **Silhouette Score**: Measures how similar objects are to their own cluster (-1 to 1; higher is better)
- **Davies-Bouldin Index**: Ratio of average distances within/between clusters (lower is better)
- **Calinski-Harabasz Score**: Ratio of between-cluster to within-cluster distances (higher is better)

## Running Tests
To verify the pipeline:
```bash
# Full pipeline execution
python main.py

# Web app test
python app.py
# Open http://127.0.0.1:5000 in browser and click "Generate Segments"
```

## Future Enhancements
- Add customer-level prediction endpoint (score new customers)
- Implement segment-specific recommendations
- Add real-time data ingestion from sales database
- Extend clustering with other algorithms (DBSCAN, spectral clustering)
- Create export functionality (CSV, JSON, reports)

## Troubleshooting

### Issue: ModuleNotFoundError
**Solution**: Ensure you're running from the project root (`C:\RFM MODEL\RFM MODEL`) and have installed dependencies:
```bash
pip install -r requirements.txt
```

### Issue: Dataset not found
**Solution**: Verify `data/Online Retail.xlsx` exists or adjust the dataset path in `src/components/data_ingestion.py`.

### Issue: Flask port already in use
**Solution**: Kill the process or specify a different port:
```bash
python app.py --port 5001
```

## Contact & Support
For questions or issues, review the logs in `logs/` directory or check the pipeline components in `src/`.

---

**Last Updated**: November 13, 2025  
**Status**: Ready for Submission
