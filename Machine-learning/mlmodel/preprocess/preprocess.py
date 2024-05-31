from fastapi import FastAPI, UploadFile, File, HTTPException,Form
import pandas as pd
from io import BytesIO
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, PowerTransformer, QuantileTransformer
from pydantic import BaseModel
import numpy as np
from sklearn.cluster import DBSCAN
from typing import Union
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from scipy.stats import zscore
from scipy.stats.mstats import winsorize 
import mysql.connector
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from category_encoders import BinaryEncoder, WOEEncoder, HashingEncoder, BaseNEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


app = FastAPI()


conn = mysql.connector.connect(
    user = "root",
    password = "root",
    host = "localhost",
    port = "3306"
)
conn.autocommit = True
cursor = conn.cursor()
database_name = 'mlpreprocessing'
create_query = f"CREATE DATABASE IF NOT EXISTS {database_name}" 
cursor.execute(create_query)
query2=f"use {database_name}"
cursor.execute(query2)
conn.commit() 
result = cursor.fetchall()
conn.commit()


def save_file_processed(df, table_name):
    create_table_query = f"CREATE TABLE IF NOT EXISTS `{table_name}` ("
    for column, dtype in df.dtypes.items():
        mysql_type = "VARCHAR(255)"  # Default to VARCHAR(255) if type is not recognized
        if dtype == 'int64':
            mysql_type = 'BIGINT'
        elif dtype == 'float64':
            mysql_type = 'FLOAT'
        elif dtype == 'object':
            # Check if it's a string or datetime
            if pd.api.types.is_string_dtype(df[column]):
                mysql_type = 'VARCHAR(255)'
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                mysql_type = 'DATETIME'
        # Adding column definition to create table query
        create_table_query += f"`{column}` {mysql_type}, "
    create_table_query = create_table_query.rstrip(', ') + ")"
    cursor.execute(create_table_query)
    # Inserting data into table
    for _, row in df.iterrows():
        # Replacing null or missing values with None
        row = row.where(pd.notnull(row), None)

        # Preparing column names and values for insertion
        columns = ', '.join([f"`{key}`" for key in row.keys()])
        placeholders = ', '.join(['%s'] * len(row))
        values = tuple(map(lambda x: x.item() if isinstance(x, np.generic) else x, row.values))
        # Inserting row into table
        insert_query = f"INSERT INTO `{table_name}` ({columns}) VALUES ({placeholders})"
        cursor.execute(insert_query, values)
    conn.commit()

def replace_special_chars_with_nan(df):
    # Define the special characters you want to replace
    special_chars = ['?', '!', '#', '@', '$', '%', '^', '&', '*', '~']

    # Iterate over each column and replace special characters with NaN
    for column in df.columns:
        df[column] = df[column].replace(special_chars,np.nan)

    return df
    

def handle_missing_values(df, method): 
    replace_special_chars_with_nan(df)
    if method == 'mean':
        numerical_imputer = SimpleImputer(strategy='mean')
        numerical_columns = df.select_dtypes(include=['number']).columns
        df[numerical_columns] = numerical_imputer.fit_transform(df[numerical_columns])
        
        # For string columns, impute using most frequent value
        string_imputer = SimpleImputer(strategy='most_frequent')
        string_columns = df.select_dtypes(include=['object']).columns
        if not string_columns.empty:
            df[string_columns] = string_imputer.fit_transform(df[string_columns])
    elif method == 'median':
        print("median..")
        # For numerical columns, impute using median
        numerical_imputer = SimpleImputer(strategy='median')
        print("numerical imputer")
        numerical_columns = df.select_dtypes(include=['number']).columns
        print("numerical columns")
        df[numerical_columns] = numerical_imputer.fit_transform(df[numerical_columns])
        print("imputerrr")
        print("numericcc",df[numerical_columns])
        # For string columns, impute using most frequent value
        string_imputer = SimpleImputer(strategy='most_frequent')
        string_columns = df.select_dtypes(include=['object']).columns
        if not string_columns.empty:
            df[string_columns] = string_imputer.fit_transform(df[string_columns])
    elif method == 'most_frequent':
    # Impute missing values using most frequent value for all columns
        imputer = SimpleImputer(strategy='most_frequent')
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        return df_imputed
    else:
        raise ValueError("Invalid method for handling missing values")

    return df
    

def scalling(df, method):
    try:
        # Selecting only numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns
        df_numeric = df[numeric_columns]

        # Initializing scaler based on the chosen method
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'min_max':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'max_abs':
            scaler = MaxAbsScaler()
        elif method == 'power':
            scaler = PowerTransformer()
        elif method == 'quantile':
            scaler = QuantileTransformer()
        else:
            raise HTTPException(status_code=400,detail="Invalid scaling method. Choose 'standard', 'min_max', 'robust', 'max_abs', 'power', or 'quantile'.")

        # Scale the numeric columns
        scaled_data = scaler.fit_transform(df_numeric)

        # Creating a DataFrame with scaled numeric columns
        df_scaled_numeric = pd.DataFrame(scaled_data, columns=df_numeric.columns)

        # Getting non-numeric columns from the original DataFrame
        non_numeric_columns = df.select_dtypes(exclude=['number']).columns
        df_non_numeric = df[non_numeric_columns]

        # Concatenating scaled numeric columns with non-numeric columns
        df_scaled = pd.concat([df_scaled_numeric, df_non_numeric], axis=1)

        # Check if there are any NaN values after scaling
        if df_scaled.isnull().values.any():
            raise HTTPException(status_code=400,detail="DataFrame contains NaN or missing values after scaling.")

        return df_scaled
    
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        raise e

# Function to detect outliers using Z-score method
def detect_outliers_zscore(data, threshold=3):
    z_scores = np.abs(zscore(data))
    outliers_mask = (z_scores > threshold).any(axis=1)
    return data[~outliers_mask]

def detect_outliers_robust_zscore(data, threshold_multiplier=3):
    median = np.median(data, axis=0)
    median_absolute_deviation = np.median(np.abs(data - median), axis=0)
    robust_z_scores = np.abs(0.6745 * (data - median) / median_absolute_deviation)
    outliers_mask = (robust_z_scores > threshold_multiplier).any(axis=1)
    return data[~outliers_mask]

def detect_outliers_iqr(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def detect_outliers_winsorization(df, lower_limit=0.15, upper_limit=0.20):
    for column in df:
        winsorized_values = winsorize(df[column], limits=(lower_limit, upper_limit))
        df[column] = winsorized_values
    return df

def detect_outliers_dbscan(data, eps=0.4, min_samples=1):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    outliers_mask = labels == -1
    return data[~outliers_mask]

def detect_outliers_isolation_forest(data, contamination=0.05):
    isolation_forest = IsolationForest(contamination=contamination)
    outliers_mask = isolation_forest.fit_predict(data) == -1
    return data[~outliers_mask]


def detect_outliers_linear_regression(data):
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(data)
    residuals = np.abs(data - pca.inverse_transform(principal_components))
    outliers_mask = (residuals > 2 * np.std(residuals)).any(axis=1)
    return data[~outliers_mask]

def detect_outliers_standard_deviation(data, threshold=3):
    try:
        std_deviation = np.std(data)
        if np.isnan(std_deviation) or std_deviation == 0:
            # If standard deviation is 0 or NaN, it returns the original data
            return data
        else:
            # Calculating the mean of the data
            mean_value = np.mean(data)

            # Creating mask to identify outliers
            outliers_mask = np.abs(data - mean_value) > threshold * std_deviation

            # Return the data without outliers
            return data[~outliers_mask]
    except Exception as e:
        return data

def detect_outliers_percentile(data, lower_percentile=5, upper_percentile=95):
    lower_limit = np.percentile(data, lower_percentile)
    upper_limit = np.percentile(data, upper_percentile)
    outliers_mask = (data < lower_limit) | (data > upper_limit)
    return data[~outliers_mask]


def outlier_removal(df, method, **kwargs):
    df_numeric = df.select_dtypes(include=[np.number])
    try:
        if method == 'zscore':
            outliers = detect_outliers_zscore(df_numeric, **kwargs)
        elif method == 'robustzscore':
            outliers = detect_outliers_robust_zscore(df_numeric, **kwargs)
        elif method == 'iqr':
            outliers = detect_outliers_iqr(df_numeric, **kwargs)
        elif method == 'winsorization':
            outliers = detect_outliers_winsorization(df, **kwargs)
        elif method == 'dbscan':
            outliers = detect_outliers_dbscan(df_numeric, **kwargs)
        elif method == 'isolation_forest':
            outliers = detect_outliers_isolation_forest(df_numeric, **kwargs)
        elif method == 'linear_regression':
            outliers = detect_outliers_linear_regression(df_numeric, **kwargs)
        elif method == 'standard_deviation':
            outliers = detect_outliers_standard_deviation(df_numeric, **kwargs)
        elif method == 'percentile':
            outliers = detect_outliers_percentile(df_numeric, **kwargs)
        else:
            raise ValueError("Invalid outlier detection method.")
        
        outlier_indices = outliers.index
        # Drop the rows that were identified as outliers
        cleaned_df = df.loc[outlier_indices]
        cleaned_data = cleaned_df.to_dict(orient='records')
        return cleaned_data
    except Exception as e:
        return str(e)

def label_encoding(df):
    encoder = LabelEncoder()
    encoded_data = df.apply(encoder.fit_transform)
    return {"encoded_data": encoded_data.to_dict(orient='records')}


def ordinal_encoding(df):
    print("starting...")
    encoder = OrdinalEncoder()
    print("started...")
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns
    print("num", non_numeric_columns)
    encoded_data = encoder.fit_transform(df[non_numeric_columns])
    print("encoded", encoded_data)
    # Get the column names after ordinal encoding
    encoded_columns = non_numeric_columns  # Use original column names
    print("encoded column",encoded_columns)
    # Convert the array to a DataFrame
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)
    print("encoded df",encoded_df)
    # Combine with the numeric columns
    df_numeric = df.select_dtypes(include=['number'])
    print("df_numeric",df_numeric)
    encoded_df = pd.concat([encoded_df, df_numeric], axis=1)
    print("encoded-df",encoded_df)
    # Convert DataFrame to dictionary
    encoded_dict = encoded_df.to_dict(orient='records')
    print("encoded dict",encoded_dict)
    return {"encoded_data": encoded_dict}

def one_hot_encoding(df):
    encoder = OneHotEncoder()
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns
    
    if len(non_numeric_columns) == 0:
        return {"encoded_data": df.to_dict(orient='records')}
    
    encoded_data = encoder.fit_transform(df[non_numeric_columns])
    # Get the column names after one-hot encoding
    encoded_columns = encoder.get_feature_names_out(input_features=non_numeric_columns)
    # Convert the CSR matrix to a DataFrame
    encoded_df = pd.DataFrame.sparse.from_spmatrix(encoded_data, columns=encoded_columns)
    # Combine with the numeric columns
    df_numeric = df.select_dtypes(include=['number'])
    encoded_df = pd.concat([encoded_df, df_numeric], axis=1)
    # Convert DataFrame to dictionary
    encoded_dict = encoded_df.to_dict(orient='records')
    return {"encoded_data": encoded_dict}

def dummy_encoding(df):
    encoder = BinaryEncoder()
    encoded_data = encoder.fit_transform(df)
    return {"encoded_data": encoded_data.to_dict(orient='records')}


def hash_encoding(df):
    encoder = HashingEncoder()
    encoded_data = encoder.fit_transform(df)
    return {"encoded_data": encoded_data.to_dict(orient='records')}


def binary_encoding(df):
    encoder = BinaryEncoder()
    encoded_data = encoder.fit_transform(df)
    return {"encoded_data": encoded_data.to_dict(orient='records')}


def base_n_encoding(df):
    encoder = BaseNEncoder()
    encoded_data = encoder.fit_transform(df)
    return {"encoded_data": encoded_data.to_dict(orient='records')}


def encoding_values(df,method):
    try:
        if method == 'label_encoding':
            encoded=label_encoding(df)
        elif method == 'one_hot_encoding':
            encoded=one_hot_encoding(df)
        elif method == 'dummy_encoding':
            encoded=dummy_encoding(df)
        elif method == 'hash_encoding':
            encoded=hash_encoding(df)
        elif method == 'binary_encoding':
            encoded = binary_encoding(df)
        elif method == 'ordinal_encoding':
            encoded = ordinal_encoding(df)
        elif method == 'base_n_encoding':
            encoded= base_n_encoding(df)
        else:
            raise ValueError("Invalid outlier detection method.") 
        return encoded
    except Exception as e:
        return str(e)

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


@app.post("/handle/")
async def handle(file: UploadFile = File(...),mode:str=Form(...)):
    print("helo")
    try:
        print("if")
        if file.filename.endswith('.csv'):
            df = read_file(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = read_excel_file(file)
        else:
            raise HTTPException(status_code=400, detail="File type not supported")
        csv_filee = file.filename.split("/")
        file_csv=csv_filee[-1].replace(".csv","_table")
        save_file_processed(df,file_csv)
        df_processed = handle_missing_values(df, mode)
        processed_table_name = f"{file_csv}_{mode}"
        save_file_processed(df_processed, processed_table_name)
        return df_processed.head(5)
    except HTTPException as http_err:
        print("httpexception")
        raise http_err
    except Exception as e :
        print("eeeexception") 
        return e    
    

@app.post("/scale/")
async def scale(file: UploadFile = File(...),mode:str=Form(...)):
    try:
        if file.filename.endswith('.csv'):
            df = read_file(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = read_excel_file(file)
        else:
            raise HTTPException(status_code=400, detail="File type not supported")
        df_numeric = df.select_dtypes(include=[np.number])
        if df_numeric.isnull().any().any():
            raise HTTPException(status_code=400, detail="Input data contains missing values (NaN)")
        
        csv_filee = file.filename.split("/")
        file_csv=csv_filee[-1].replace(".csv","_table")
        save_file_processed(df,file_csv)
        df_processed = scalling(df,mode)
        
        processed_table_name = f"{file_csv}_{mode}"
        save_file_processed(df_processed, processed_table_name)
        return df_processed.head(5)
    
    except HTTPException as http_err:
        raise http_err
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")



@app.post("/outlier/")
async def outlier(file: UploadFile = File(...),mode:str=Form(...)):
    try:
        if file.filename.endswith('.csv'):
            df = read_file(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = read_excel_file(file)
        else:
            raise HTTPException(status_code=400, detail="File type not supported")
        df_numeric = df.select_dtypes(include=[np.number]) 
        if df_numeric.isnull().any().any():
            raise HTTPException(status_code=400, detail="Input data contains missing values (NaN)")
        string_columns = df.select_dtypes(include=['object']).columns
        if len(string_columns) > 0:
            raise HTTPException(status_code=400, detail="Data contains string values.")     
       
        csv_filee = file.filename.split("/")
        file_csv=csv_filee[-1].replace(".csv","_table")
        save_file_processed(df,file_csv)

        df_processed_dict= outlier_removal(df, mode)
        df_processed = pd.DataFrame(df_processed_dict, columns=df.columns)

        processed_table_name = f"{file_csv}_{mode}"
        save_file_processed(df_processed, processed_table_name)
 
        return df_processed_dict

    except HTTPException as http_err:
        raise http_err
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/encoding")
async def encodingStringValues(file: UploadFile = File(...),mode:str=Form(...)):
    try:
        if file.filename.endswith('.csv'):
          df = read_file(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
          df = read_excel_file(file)
        else:
          raise HTTPException(status_code=400, detail="File type not supported")
        csv_filee = file.filename.split("/")

        if df.isnull().values.any():
            raise HTTPException(status_code=400, detail="Data contains null values.")

        print("dff",df)
        file_csv=csv_filee[-1].replace(".csv","_table")
        save_file_processed(df, file_csv)
        df_encoded=encoding_values(df,mode)
        processed_table_name = f"{file_csv}_{mode}"
      
        df_processed = pd.DataFrame(df_encoded['encoded_data'],columns=df_encoded['encoded_data'][0].keys())
        print("df_processed",df_processed)
        save_file_processed(df_processed, processed_table_name)
        return df_encoded
    except HTTPException as http_err:
        raise http_err
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


    



