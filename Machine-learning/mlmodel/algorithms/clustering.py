from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from sklearn.cluster import KMeans,MeanShift,DBSCAN
import numpy as np
import io

app = FastAPI()

class ClusteredData(BaseModel):
    data: list = Field(..., description="Clustered data in the form of list of dictionaries.")
    cluster_labels: list = Field(..., description="Cluster labels assigned by KMeans algorithm.")

def perform_kmeans_clustering(data: pd.DataFrame, n_clusters: int = 3):
    try:
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters)
        cluster_labels = kmeans.fit_predict(data)

        # Add cluster labels to the DataFrame
        data['cluster'] = cluster_labels

        # Convert DataFrame to list of dictionaries
        clustered_data = data.to_dict(orient='records')

        return ClusteredData(data=clustered_data, cluster_labels=cluster_labels.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/kmeans")
async def kmeans_clustering(file: UploadFile, n_clusters: int = 3):
    content = await file.read()
    # Use io.BytesIO to convert the content to a file-like object
    file_like_object = io.BytesIO(content)

    # Read the data from the file-like object
    if file.filename.endswith('.csv'):
        data = pd.read_csv(file_like_object)
    elif file.filename.endswith(('.xls', '.xlsx')):
        data = pd.read_excel(file_like_object)
    else:
        raise HTTPException(status_code=400, detail="File type not supported")

    # Perform K-means clustering
    clustered_data = perform_kmeans_clustering(data, n_clusters)

    return clustered_data

def perform_mean_shift_clustering(data: pd.DataFrame, bandwidth: float = None):
    try:
        # Perform Mean Shift clustering
        ms = MeanShift(bandwidth=bandwidth)
        cluster_labels = ms.fit_predict(data)

        # Add cluster labels to the DataFrame
        data['cluster'] = cluster_labels

        # Convert DataFrame to list of dictionaries
        clustered_data = data.to_dict(orient='records')

        return ClusteredData(data=clustered_data, cluster_labels=cluster_labels.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/meanshift")
async def meanshift_clustering(file: UploadFile, bandwidth: float = None):
    content = await file.read()
    # Use io.BytesIO to convert the content to a file-like object
    file_like_object = io.BytesIO(content)

    # Read the data from the file-like object
    if file.filename.endswith('.csv'):
        data = pd.read_csv(file_like_object)
    elif file.filename.endswith(('.xls', '.xlsx')):
        data = pd.read_excel(file_like_object)
    else:
        raise HTTPException(status_code=400, detail="File type not supported")

    # Perform Mean Shift clustering
    clustered_data = perform_mean_shift_clustering(data, bandwidth)

    return clustered_data

def perform_dbscan_clustering(df: pd.DataFrame, eps: float = 0.5, min_samples: int = 5):
    try:
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(df)

        print("Cluster Labels:", cluster_labels)  # Add this line to print cluster labels

        # Combine data with cluster labels
        clustered_data = []
        for i, data_point in enumerate(df.to_dict(orient='records')):
            data_point['cluster_label'] = int(cluster_labels[i])  # Ensure cluster label is converted to int
            clustered_data.append(data_point)

        return clustered_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")




@app.post("/dbscan/")
async def dbscan_clustering(file: UploadFile, eps: float = 0.5, min_samples: int = 5):
    content = await file.read()
    # Use io.BytesIO to convert the content to a file-like object
    file_like_object = io.BytesIO(content)

    # Read the data from the file-like object
    if file.filename.endswith('.csv'):
        data = pd.read_csv(file_like_object)
    elif file.filename.endswith(('.xls', '.xlsx')):
        data = pd.read_excel(file_like_object)
    else:
        raise HTTPException(status_code=400, detail="File type not supported")

    # Perform DBSCAN clustering
    clustered_data = perform_dbscan_clustering(data, eps, min_samples)

    return clustered_data
