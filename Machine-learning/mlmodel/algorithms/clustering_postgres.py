from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from sklearn.cluster import KMeans,MeanShift,DBSCAN
import numpy as np
from datetime import datetime
from io import BytesIO 
import psycopg2

app = FastAPI()

database_name = 'mlproj'

conn = psycopg2.connect(
    database = database_name,
    user="postgres",
    password="Sbksatya@1919",
    host="localhost",
    port="5432"
)
conn.autocommit = True
cursor = conn.cursor()

def save_file_processed(df, table_name):
    create_table_query = f"CREATE TABLE IF NOT EXISTS \"{table_name}\" ("
    for column, dtype in df.dtypes.items():
        postgresql_type = "VARCHAR(255)"  
        if dtype == 'int64':
            postgresql_type = 'BIGINT'
        elif dtype == 'float64':
            postgresql_type = 'FLOAT'
        elif dtype == 'object':
            # Check if it's a string or datetime
            if pd.api.types.is_string_dtype(df[column]):
                postgresql_type = 'VARCHAR(255)'
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                postgresql_type = 'TIMESTAMP'
        # Adding column definition to create table query
        create_table_query += f"\"{column}\" {postgresql_type}, "
        print("creating..")
    create_table_query = create_table_query.rstrip(', ') + ")"
    cursor.execute(create_table_query)
    print("created..")
    # Inserting data into table
    for _, row in df.iterrows():
        # Replacing null or missing values with None
        row = row.where(pd.notnull(row), None)

        # Preparing column names and values for insertion
        columns = ', '.join([f"\"{key}\"" for key in row.keys()])
        placeholders = ', '.join(['%s'] * len(row))
        values = tuple(map(lambda x: x.item() if isinstance(x, np.generic) else x, row.values))
        # Inserting row into table
        insert_query = f"INSERT INTO \"{table_name}\" ({columns}) VALUES ({placeholders})"
        cursor.execute(insert_query, values)
        print("inserting...")
    conn.commit()


def format_time(build_time):
    build_time_str = str(build_time)
    build_time_parts = build_time_str.split(':')
    hours = build_time_parts[0]
    minutes = build_time_parts[1]
    seconds_microseconds = build_time_parts[2].split('.')
    seconds = seconds_microseconds[0]
    milliseconds = seconds_microseconds[1]
    formatted_build_time = f"{hours}:{minutes}:{seconds}:{milliseconds}"
    return formatted_build_time


def read_file(file: UploadFile):
    try:
        contents = file.file.read()
        return pd.read_csv(BytesIO(contents),encoding='ISO-8859-1')
    except Exception as e:
        return None

def read_excel_file(file: UploadFile):
    try:
        contents = file.file.read()
        return pd.read_excel(BytesIO(contents))
    except Exception as e:
        print(f"An error occurred while reading the Excel file: {e}")
        return None
    
table_name=' '


@app.post("/source")
async def source(file: UploadFile = File(...)):
    try:
        if file.filename.endswith('.csv'):
            df = read_file(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = read_excel_file(file)
        else:
            raise HTTPException(status_code=400, detail="File type not supported")
        csv_filee = file.filename.split("/")
        file_csv=csv_filee[-1].replace(".csv","_table")
        file_csv = file_csv[0].lower() + file_csv[1:]
        global table_name
        table_name=file_csv
        save_file_processed(df,file_csv)
        return "success"
    except HTTPException as http_err:
        print("httpexception") 
        raise http_err
    except Exception as e :
        print("eeeexception") 
        return e  

class ClusteredData(BaseModel):
    data: list = Field(..., description="Clustered data in the form of list of dictionaries.")
    cluster_labels: list = Field(..., description="Cluster labels assigned by KMeans algorithm.")

@app.post("/kmeans")
async def kmeans_clustering(n_clusters: int = 3):
    
    try:
        start_time=datetime.now()
        global table_name
        query = f"SELECT * FROM {table_name}"
        data = pd.read_sql_query(query, conn)


        kmeans = KMeans(n_clusters=n_clusters)
        cluster_labels = kmeans.fit_predict(data)

        # Add cluster labels to the DataFrame
        data['cluster'] = cluster_labels

        # Convert DataFrame to list of dictionaries
        clustered_data = data.to_dict(orient='records')
        end_time=datetime.now()
        build_time=end_time-start_time
        formatted_build_time=format_time(build_time)

        return ClusteredData(data=clustered_data, cluster_labels=cluster_labels.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/meanshift")
async def meanshift_clustering( bandwidth: float = None):
    try:
        
        global table_name
        query = f"SELECT * FROM {table_name}"
        data = pd.read_sql_query(query, conn)


        ms = MeanShift(bandwidth=bandwidth)
        cluster_labels = ms.fit_predict(data)

        # Add cluster labels to the DataFrame
        data['cluster'] = cluster_labels

        # Convert DataFrame to list of dictionaries
        clustered_data = data.to_dict(orient='records')

        return ClusteredData(data=clustered_data, cluster_labels=cluster_labels.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/dbscan/")
async def dbscan_clustering(eps: float = 0.5, min_samples: int = 5):
    try:
        global table_name
        query = f"SELECT * FROM {table_name}"
        data = pd.read_sql_query(query, conn)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(data)

        print("Cluster Labels:", cluster_labels)  # Add this line to print cluster labels

        # Combine data with cluster labels
        clustered_data = []
        for i, data_point in enumerate(data.to_dict(orient='records')):
            data_point['cluster_label'] = int(cluster_labels[i])  # Ensure cluster label is converted to int
            clustered_data.append(data_point)

        return clustered_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    
