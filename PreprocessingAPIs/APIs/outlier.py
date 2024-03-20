import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor
from scipy.stats import zscore, iqr, trim_mean, scoreatpercentile
import matplotlib.pyplot as plt
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
# Load data from CSV file



# Function to detect outliers using Z-score method
def detect_outliers_zscore(data, threshold=3):
    z_scores = np.abs(zscore(data))
    outliers_mask = (z_scores > threshold).any(axis=1)
    return data[~outliers_mask]

# Function to detect outliers using Robust Z-score
def detect_outliers_robust_zscore(data, threshold=3):
    median = np.median(data, axis=0)
    median_absolute_deviation = np.median(np.abs(data - median), axis=0)
    robust_z_scores = np.abs(0.6745 * (data - median) / median_absolute_deviation)
    outliers_mask = (robust_z_scores > threshold).any(axis=1)
    return data[~outliers_mask]

# Function to detect outliers using I.Q.R method
def detect_outliers_iqr(data, k=1.5):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr_values = iqr(data)
    lower_bound = q1 - k * iqr_values
    upper_bound = q3 + k * iqr_values
    outliers_mask = ((data < lower_bound) | (data > upper_bound)).any(axis=1)
    return data[~outliers_mask]

# Function to detect outliers using Winsorization method (Percentile Capping)
def detect_outliers_winsorization(data, percentile=5):
    lower_limit = scoreatpercentile(data, percentile)
    upper_limit = scoreatpercentile(data, 100 - percentile)
    data = np.where(data < lower_limit, lower_limit, data)
    data = np.where(data > upper_limit, upper_limit, data)
    return data

# Function to detect outliers using DBSCAN Clustering
def detect_outliers_dbscan(data, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    outliers_mask = labels == -1
    return data[~outliers_mask]

# Function to detect outliers using Isolation Forest
def detect_outliers_isolation_forest(data, contamination=0.05):
    isolation_forest = IsolationForest(contamination=contamination)
    outliers_mask = isolation_forest.fit_predict(data) == -1
    return data[~outliers_mask]

# Function to detect outliers using Linear Regression Models (PCA, LMS)
def detect_outliers_linear_regression(data):
    pca = PCA(n_components=1)
    principal_components = pca.fit_transform(data)
    residuals = np.abs(data - pca.inverse_transform(principal_components))
    outliers_mask = (residuals > 2 * np.std(residuals)).any(axis=1)
    return data[~outliers_mask]

# Function to detect outliers using Standard Deviation
def detect_outliers_standard_deviation(data, threshold=3):
    std_deviation = np.std(data)
    outliers_mask = np.abs(data - np.mean(data)) > threshold * std_deviation
    return data[~outliers_mask]

# Function to detect outliers using Percentile
def detect_outliers_percentile(data, lower_percentile=5, upper_percentile=95):
    lower_limit = np.percentile(data, lower_percentile)
    upper_limit = np.percentile(data, upper_percentile)
    outliers_mask = (data < lower_limit) | (data > upper_limit)
    return data[~outliers_mask]

# Function to visualize the data
def visualize_data(data):
    # Add code here to visualize the data (scatter plots, box plots, histograms, etc.)
    pass  # Placeholder implementation




# Save data without outliers to new CSV files (if needed)
# data_without_outliers_hypothesis_testing.to_csv('data_without_outliers_hypothesis_testing.csv', index=False)
# data_without_outliers_zscore.to_csv('data_without_outliers_zscore.csv', index=False)
# data_without_outliers_robust_zscore.to_csv('data_without_outliers_robust_zscore.csv', index=False)
# data_without_outliers_iqr.to_csv('data_without_outliers_iqr.csv', index=False)
# data_without_outliers_winsorization.to_csv('data_without_outliers_winsorization.csv', index=False)
# data_without_outliers_dbscan.to_csv('data_without_outliers_dbscan.csv', index=False)
# data_without_outliers_isolation_forest.to_csv('data_without_outliers_isolation_forest.csv', index=False)
# data_without_outliers_linear_regression.to_csv('data_without_outliers_linear_regression.csv', index=False)
# data_without_outliers_standard_deviation.to_csv('data_without_outliers_standard_deviation.csv', index=False)
# data_without_outliers_percentile.to_csv('data_without_outliers_percentile.csv', index=False)
app = FastAPI()

class CSV_file(BaseModel):
    path : str
    mode : str
    delimit : str = ','
def outlier_removal(df: pd.DataFrame, method,**kwargs):
    if method == 'zscore':
        outliers = detect_outliers_zscore(df, **kwargs)
    elif method == 'robust_zscore':
        outliers = detect_outliers_robust_zscore(df, **kwargs)
    elif method == 'iqr':
        outliers = detect_outliers_iqr(df, **kwargs)
    elif method == 'winsorization':
        outliers = detect_outliers_winsorization(df, **kwargs)
    elif method == 'dbscan':
        outliers = detect_outliers_dbscan(df, **kwargs)
    elif method == 'isolation_forest':
        outliers = detect_outliers_isolation_forest(df, **kwargs)
    elif method == 'linear_regression':
        outliers = detect_outliers_linear_regression(df, **kwargs)
    elif method == 'standard_deviation':
        outliers = detect_outliers_standard_deviation(df, **kwargs)
    elif method == 'percentile':
        outliers = detect_outliers_percentile(df, **kwargs)
    else:
        raise ValueError("Invalid outlier detection method.")

    return outliers
@app.post("/outlier/")
async def outlier(csv_file : CSV_file):
    print("strt")
    try:
        if csv_file.path.endswith('.csv'):
            df = pd.read_csv(csv_file.path, delimiter=csv_file.delimit)
            print("In IF Block")
            print(df)
            # return df.head(10)
        elif csv_file.path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(csv_file.path)
        else:
            raise HTTPException(status_code=400, detail="File type not supported")
        print("handling")
        df_processed = outlier_removal(df, csv_file.mode)
        print(df_processed)
        # with BytesIO() as buffer:
        #     df_processed.to_csv(buffer, index=False)
        #     buffer.seek(0)
            # return FileResponse(buffer, filename="processed_data.csv")
        return df_processed.head(5)
    except Exception as e : 
        return e