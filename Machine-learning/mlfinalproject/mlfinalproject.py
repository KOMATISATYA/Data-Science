from fastapi import FastAPI, UploadFile, File, HTTPException,Form
import pandas as pd
from io import BytesIO
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, PowerTransformer, QuantileTransformer
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from scipy.stats import zscore
from scipy.stats.mstats import winsorize 
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from fastapi import FastAPI,UploadFile,File,Form,HTTPException
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_classif,r_regression,mutual_info_classif,mutual_info_regression
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.svm import SVR,SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import psycopg2
from io import BytesIO
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score,roc_auc_score,log_loss,average_precision_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
import xgboost as xgb
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_squared_log_error, explained_variance_score, median_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report,r2_score
from sklearn.svm import SVR
from sklearn.preprocessing import OrdinalEncoder
from fastapi.middleware.cors import CORSMiddleware
from typing import List


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4000", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

database_name = 'mlfinal'

conn = psycopg2.connect(
    database = database_name,
    user="postgres",
    password="postgres",
    host="localhost",
    port="5432"
)
conn.autocommit = True
cursor = conn.cursor()

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

def save_file_processed(df, table_name):
    create_table_query = f"CREATE TABLE IF NOT EXISTS \"{table_name}\" ("
    for column, dtype in df.dtypes.items():
        postgresql_type = "VARCHAR(255)"  # Default to VARCHAR(255) if type is not recognized
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
    print(data)
    print(data[~outliers_mask])
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
    encoded_columns = non_numeric_columns  
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


def encoding_values(df,method):
    try:
        if method == 'label_encoding':
            encoded=label_encoding(df)
        elif method == 'ordinal_encoding':
            encoded=ordinal_encoding(df)
        else:
            raise ValueError("Invalid encoding method detection method.") 
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
        raise http_err
    except Exception as e :
        return e  

  
@app.post("/handle")    
async def handle(mode:str=Form(...)):
    try:
      global table_name
      if table_name==' ':
        raise HTTPException(status_code=400, detail="First Upload source csv file")
      
      query=f"select * from {table_name}"
      df = pd.read_sql_query(query, conn)
      df.replace(to_replace=[None], value=np.nan, inplace=True)
      df_processed = handle_missing_values(df, mode)
      processed_table_name = f"{table_name}_{mode}"
      table_name=processed_table_name
      save_file_processed(df_processed, processed_table_name)
      return "success"
    except HTTPException as http_err:
        print("httpexception") 
        raise http_err
    except Exception as e :
        print("eeeexception") 
        return e  


@app.post("/remove_duplicates")    
async def remove_duplicates():
    try:
        global table_name
        if table_name==' ':
            raise HTTPException(status_code=400, detail="First Upload source csv file")
      
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        
        # Remove duplicates
        df.drop_duplicates(inplace=True)
        print(df)
        processed_table_name = f"{table_name}_removed_duplicates"
        print(processed_table_name)
        table_name = processed_table_name
        save_file_processed(df, processed_table_name)
        return "success"
    except HTTPException as http_err:
        print("HTTPException:", http_err) 
        raise http_err
    except Exception as e:
        print("Exception:", e) 
        return str(e)


@app.post("/scale")    
async def scale(mode:str=Form(...), target_column: str = Form(...)):
    try:
      global table_name
      if table_name==' ':
        raise HTTPException(status_code=400, detail="First Upload source csv file")
      
      query=f"select * from {table_name}"
      df = pd.read_sql_query(query, conn)
      df_numeric = df.select_dtypes(include=[np.number])
      if df_numeric.isnull().any().any():
        raise HTTPException(status_code=400, detail="Input data contains missing values (NaN)")
      string_columns = df.select_dtypes(include=['object']).columns
      if len(string_columns) > 0:
        raise HTTPException(status_code=400, detail="Data contains string values.") 
    
      X = df.drop(columns=[target_column])
      y = df[target_column]
      df_processed = scalling(X,mode)
      df_processed=pd.concat([df_processed,y],axis=1)
      processed_table_name = f"{table_name}_{mode}"
      table_name=processed_table_name
      save_file_processed(df_processed, processed_table_name)
      return "success"
    except HTTPException as http_err:
        print("httpexception") 
        raise http_err
    except Exception as e :
        print("eeeexception") 
        return e  


@app.post("/encoding")    
async def encoding(mode:str=Form(...)):
    try:
      global table_name
      if table_name==' ':
        raise HTTPException(status_code=400, detail="First Upload source csv file")
      
      query=f"select * from {table_name}"
      df = pd.read_sql_query(query, conn)
      df_numeric = df.select_dtypes(include=[np.number])
      if df_numeric.isnull().any().any():
         raise HTTPException(status_code=400, detail="Input data contains missing values (NaN)")
    
      df_encoded=encoding_values(df,mode)
      processed_table_name = f"{table_name}_{mode}"
      table_name=processed_table_name
      df_processed = pd.DataFrame(df_encoded['encoded_data'],columns=df_encoded['encoded_data'][0].keys())
      save_file_processed(df_processed, processed_table_name)
      return "success"
    except HTTPException as http_err:
        raise http_err
    except Exception as e :
        return e  


@app.post("/outlier")    
async def outlier(mode:str=Form(...)):
    try:
      global table_name
      if table_name==' ':
        raise HTTPException(status_code=400, detail="First Upload source csv file")
      
      query=f"select * from {table_name}"
      df = pd.read_sql_query(query, conn)
      df_numeric = df.select_dtypes(include=[np.number])
      if df_numeric.isnull().any().any():
        raise HTTPException(status_code=400, detail="Input data contains missing values (NaN)")
      string_columns = df.select_dtypes(include=['object']).columns
      if len(string_columns) > 0:
        raise HTTPException(status_code=400, detail="Data contains string values.")
      df_processed_dict= outlier_removal(df, mode)
      df_processed = pd.DataFrame(df_processed_dict, columns=df.columns)
      processed_table_name = f"{table_name}_{mode}"
      table_name=processed_table_name
      save_file_processed(df_processed, processed_table_name)
      return "success"
    except HTTPException as http_err:
        raise http_err
    except Exception as e :
        return e  


@app.post("/variance_filter")
async def variance_threshold_selector(
      threshold: float = Form(0.0),target_column: str = Form(...)):
    try:
        global table_name
        if table_name==' ':
            raise HTTPException(status_code=400, detail="First Upload source csv file")
      
        print(table_name)
        query = f"SELECT * FROM {table_name}"
        data = pd.read_sql_query(query, conn)

        if target_column not in data.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data.")

        string_columns = data.select_dtypes(include=['object']).columns
        if len(string_columns) > 0:
            raise HTTPException(status_code=400, detail="Data contains string values. Please encode the data first.")

        X = data.drop(columns=[target_column])
        y = data[target_column]

        selector = VarianceThreshold(threshold)
        selector.fit(X)

        # Get the selected features
        selected_features = X.iloc[:, selector.get_support(indices=True)]
        # Concatenate the selected features with the target column
        selected_data = pd.concat([selected_features, y], axis=1)

        # Define the name for the processed table
        processed_table_name = f"{table_name}_variance"
        
        # Save the selected data to the database table
        save_file_processed(selected_data, processed_table_name)

        return selected_features.columns.tolist()
    except Exception as e:
        raise HTTPException(status_code=200, detail=str(e))


@app.post("/correlation_selector")
async def correlation_selector(
      threshold: float = Form(0.0),target_column: str = Form(...)):
    try:
        global table_name
        if table_name==' ':
            raise HTTPException(status_code=400, detail="First Upload source csv file")
      
        query = f"SELECT * FROM {table_name}"
        data = pd.read_sql_query(query, conn)

        if target_column not in data.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data.")

        string_columns = data.select_dtypes(include=['object']).columns
        if len(string_columns) > 0:
            raise HTTPException(status_code=400, detail="Data contains string values. Please encode the data first.")

        X = data.drop(columns=[target_column])
        y = data[target_column]

        cor = X.corrwith(y)
        selected_features = X.loc[:, abs(cor) > threshold]

        # Concatenate the selected features with the target column
        selected_data = pd.concat([selected_features, y], axis=1)

        # Define the name for the processed table
        processed_table_name = f"{table_name}_correlation"
        
        # Save the selected data to the database table
        save_file_processed(selected_data, processed_table_name)

        return selected_features.columns.tolist()
    except Exception as e:
        raise HTTPException(status_code=200, detail=str(e))
    

@app.post("/univariate_selector")
async def univariate_selector(
      k: int = Form(10),
      method: str = Form("f_classif"),target_column: str = Form(...)):
    try:
        global table_name
        if table_name==' ':
            raise HTTPException(status_code=400, detail="First Upload source csv file")
      
        print(table_name)
        query = f"SELECT * FROM {table_name}"
        data = pd.read_sql_query(query, conn)

        if target_column not in data.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data.")

        if data.isnull().values.any():
            raise HTTPException(status_code=400, detail="Data contains null values. Please perform missing values preprocessing step")

        string_columns = data.select_dtypes(include=['object']).columns
        if len(string_columns) > 0:
            raise HTTPException(status_code=400, detail="Data contains string values. Please encode the data first.")

        X = data.drop(columns=[target_column])
        y = data[target_column]

        if method == 'chi2':
            selector = SelectKBest(chi2, k=k)
        elif method == 'f_classif':
            selector = SelectKBest(f_classif, k=k)
        elif method == 'mutual_info_classif':
            selector = SelectKBest(mutual_info_classif, k=k)
        elif method == 'r_regression':
            selector = SelectKBest(r_regression, k=k)
        elif method == 'mutual_info_regression':
            selector = SelectKBest(mutual_info_regression, k=k)
        selector.fit(X, y)

        selected_features = X.columns[selector.get_support(indices=True)].tolist()

        # Concatenate the selected features with the target column
        selected_data = pd.concat([data[selected_features], y], axis=1)

        # Define the name for the processed table
        processed_table_name = f"{table_name}_univariate"

        # Save the selected data to the database table
        save_file_processed(selected_data, processed_table_name)

        return selected_features
    except Exception as e:
        raise HTTPException(status_code=200, detail=str(e))


def get_estimator(method):
    if method == "linear_regression":
        return LinearRegression()
    elif method == "random_forest_regressor":
        return RandomForestRegressor()
    elif method == "random_forest_classifier":
        return RandomForestClassifier()
    else:
        raise HTTPException(status_code=400, detail="Invalid estimator method")

@app.post("/recursive_feature")
async def recursive_feature_elimination(
      n: int = Form(...),
      method: str = Form(...),target_column: str = Form(...)):
    try:
       
        global table_name
        if table_name==' ':
            raise HTTPException(status_code=400, detail="First Upload source csv file")
      
        print(table_name)
        query = f"SELECT * FROM {table_name}"
        data = pd.read_sql_query(query, conn)

        if target_column not in data.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data.")

        if data.isnull().values.any():
            raise HTTPException(status_code=400, detail="Data contains null values. Please perform missing values preprocessing step")

        string_columns = data.select_dtypes(include=['object']).columns
        if len(string_columns) > 0:
            raise HTTPException(status_code=400, detail="Data contains string values. Please encode the data first.")

        estimator = get_estimator(method)
        rfe = RFE(estimator=estimator, n_features_to_select=n)

        X = data.drop(columns=[target_column])
        y = data[target_column]

        rfe.fit(X, y)

        selected_features = X.columns[rfe.get_support(indices=True)].tolist()

        # Concatenate the selected features with the target column
        selected_data = pd.concat([data[selected_features], y], axis=1)

        # Define the name for the processed table
        processed_table_name = f"{table_name}_rfe"

        # Save the selected data to the database table
        save_file_processed(selected_data, processed_table_name)

        return selected_features
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/lasso_feature_selection")
async def lasso_feature_selection(
    alpha: float = Form(...),target_column: str = Form(...)
):
    try:
        global table_name
        if table_name==' ':
            raise HTTPException(status_code=400, detail="First Upload source csv file")
        print(table_name)
        query = f"SELECT * FROM {table_name}"
        data = pd.read_sql_query(query, conn)

        if target_column not in data.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data.")

        if data.isnull().values.any():
            raise HTTPException(status_code=400, detail="Data contains null values. Please perform missing values preprocessing step")

        string_columns = data.select_dtypes(include=['object']).columns
        if len(string_columns) > 0:
            raise HTTPException(status_code=400, detail="Data contains string values. Please encode the data first.")

        # Split features and target
        X = data.drop(columns=[target_column])  # Assuming the target column is named "target"
        y = data[target_column]

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform Lasso feature selection
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_scaled, y)

        # Extract selected features based on non-zero coefficients
        selected_features_indices = [i for i, coef in enumerate(lasso.coef_) if coef != 0]
        selected_features = X.columns[selected_features_indices].tolist()

        # Concatenate the selected features with the target column
        selected_data = pd.concat([data[selected_features], y], axis=1)

        # Define the name for the processed table
        processed_table_name = f"{table_name}_lasso"

        # Save the selected data to the database table
        save_file_processed(selected_data, processed_table_name)

        return {"selected_features": selected_features}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pca")
async def perform_pca(target_column: str = Form(...)):
  
    global table_name
    if table_name==' ':
        raise HTTPException(status_code=400, detail="First Upload source csv file")
      
    
    print(table_name)
    query = f"SELECT * FROM {table_name}"
    data = pd.read_sql_query(query, conn)

    if target_column not in data.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data.")
    
    string_columns = data.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        raise HTTPException(status_code=400, detail="Data contains string values. please do encoing step first")
    
    if data.isnull().values.any():
        raise HTTPException(status_code=400, detail="Data contains null values. Please perform missing values preprocessing step")
    
     
    X = data.drop(columns=[target_column])  # Assuming the target column is named "target"
    y = data[target_column]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Exclude target column
    pca = PCA()
    pca.fit(X_scaled)
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    pca = PCA(n_components=num_components)
    X_pca = pca.fit_transform(X_scaled)

        # Concatenate the transformed features with the target column
    transformed_data = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(num_components)])
    transformed_data[target_column] = y

        # Define the name for the processed table
    processed_table_name = f"{table_name}_pca"
    table_name=processed_table_name
        # Save the combined and transformed data to the database table
    save_file_processed(transformed_data, processed_table_name)

    return {"transformed_features": X_pca.tolist()}


@app.post("/decision-tree-classification")
async def decision_tree_classification(
    criterion: str = Form("gini", description="The function to measure the quality of a split"),
    max_depth: int = Form(None, description="The maximum depth of the tree"),
    min_samples_split: int = Form(2, description="The minimum number of samples required to split an internal node"),
    min_samples_leaf: int = Form(1, description="The minimum number of samples required to be at a leaf node"),
    max_leaf_nodes: int = Form(None, description="Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity."),
    target_column: str = Form(...),
    independent_variables: List[str] = Form(...),
    train_percent: float = Form(...), 
    test_percent: float = Form(...), 
    validation_percent: float = Form(...)
):
    try:
        start_time = datetime.now()
    
        global table_name
        if table_name==' ':
            raise HTTPException(status_code=400, detail="First Upload source csv file")
        
        print(table_name)
       
        
        independent_variables_split = [f'"{col.strip()}"' for col in independent_variables[0].split(',')]
        independent_variables_joined = ', '.join(independent_variables_split)
        query = f"SELECT \"{target_column}\", {independent_variables_joined} FROM \"{table_name}\""

        data = pd.read_sql_query(query, conn)


        if target_column not in data.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data.")

        if data.isnull().values.any():
            raise HTTPException(status_code=400, detail="Data contains null values.")
        
        string_columns = data.select_dtypes(include=['object']).columns
        if len(string_columns) > 0:
            raise HTTPException(status_code=400, detail="Data contains string values.")

    
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Split the dataset into training and testing sets
            
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size= train_percent, random_state=42)
        validation_size = validation_percent / (test_percent + validation_percent)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_size, random_state=42)
        
        # Define parameter grid for GridSearchCV
        param_grid = {
            'criterion': [criterion],
            'max_depth': [max_depth],
            'min_samples_split': [min_samples_split],
            'min_samples_leaf': [min_samples_leaf],
            'max_leaf_nodes': [max_leaf_nodes]
        }
        
        # Initialize DecisionTreeClassifier
        clf = DecisionTreeClassifier(random_state=42)
        
        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        
        # Fit GridSearchCV on training data
        grid_search.fit(X_train, y_train)
        
        # Get the best estimator
        best_estimator = grid_search.best_estimator_
        
        # Make predictions on the test set
        y_pred = best_estimator.predict(X_test)
        y_pred_proba = best_estimator.predict_proba(X_test)  # For Log Loss and AUC

        unique_classes = len(np.unique(y_test)) 
        if unique_classes == 2:  
            avg = 'binary'
        else: 
            avg = 'macro'
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=avg)
        recall = recall_score(y_test, y_pred, average=avg)  
        f1 = f1_score(y_test, y_pred, average=avg)
        f1_micro = f1_score(y_test, y_pred, average='micro')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        recall_micro = recall_score(y_test, y_pred, average='micro')
        recall_macro = recall_score(y_test, y_pred, average='macro')
        recall_weighted = recall_score(y_test, y_pred, average='weighted')
        logloss = log_loss(y_test, y_pred_proba)  
        
        # Confusion Matrix and Specificity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        confusion_mat = confusion_matrix(y_test, y_pred).tolist()
        specificity = tn / (tn + fp)  
        sensitivity = recall 
        # Calculate AUC
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        balanced_accuracy = (specificity + sensitivity) / 2
        average_precision = average_precision_score(y_test, y_pred_proba[:, 1])
        end_time=datetime.now()

        build_time=end_time-start_time
        formatted_build_time=format_time(build_time)

        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall (sensitivity)": sensitivity,
            "specificity": specificity,
            "f1_score": f1,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "recall_micro": recall_micro,
            "recall_macro": recall_macro,
            "recall_weighted": recall_weighted,
            "log_loss": logloss,
            "auc": auc,
            "balanced_accuracy":balanced_accuracy,
            "average_precision":average_precision,
            "confusion_matrix":confusion_mat,
            "build_time":formatted_build_time,
            "best_params": grid_search.best_params_,
            
        }
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/random-forest-classification")
async def randomForest(
                 estimators:int=Form(...,description="number of decision tree"),
                 criterion:str=Form(...,description="The function to measure the quality of a split"),
                 target_column: str = Form(...),
                 independent_variables: List[str] = Form(...),
                 train_percent: float = Form(...), 
                test_percent: float = Form(...), 
                validation_percent: float = Form(...)):
    start_time=datetime.now()
      
    global table_name
    
    if table_name==' ':
        raise HTTPException(status_code=400, detail="First Upload source csv file")
    print(table_name)

    
    independent_variables_split = [f'"{col.strip()}"' for col in independent_variables[0].split(',')]
    independent_variables_joined = ', '.join(independent_variables_split)
    query = f"SELECT \"{target_column}\", {independent_variables_joined} FROM \"{table_name}\""

    data = pd.read_sql_query(query, conn)


    if target_column not in data.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data.")

    if data.isnull().values.any():
        raise HTTPException(status_code=400, detail="Data contains null values.")
    
    string_columns = data.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        raise HTTPException(status_code=400, detail="Data contains string values.")


    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split the dataset into training and testing sets
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size= train_percent, random_state=42)
    validation_size = validation_percent / (test_percent + validation_percent)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_size, random_state=42)


    clf = RandomForestClassifier(n_estimators=estimators, criterion=criterion)
    clf.fit(X_train, y_train)
  
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)  
    unique_classes = len(np.unique(y_test)) 
    if unique_classes == 2: 
        avg = 'binary'
    else:
        avg = 'macro'
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=avg)
    recall = recall_score(y_test, y_pred, average=avg)  
    f1 = f1_score(y_test, y_pred, average=avg)
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    recall_micro = recall_score(y_test, y_pred, average='micro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    logloss = log_loss(y_test, y_pred_proba) 
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    confusion_mat = confusion_matrix(y_test, y_pred).tolist()
    specificity = tn / (tn + fp) 
    sensitivity = recall  
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    sensitivity = recall
    balanced_accuracy = (specificity + sensitivity) / 2
    average_precision = average_precision_score(y_test, y_pred_proba[:, 1])

    end_time=datetime.now()
    build_time=end_time-start_time
    formatted_build_time=format_time(build_time)

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall (sensitivity)": sensitivity,
        "specificity": specificity,
        "f1_score": f1,
         "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "recall_micro": recall_micro,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "log_loss": logloss,
        "auc": auc,
        "balanced_accuracy":balanced_accuracy,
        "average-precision":average_precision,
        "confusion_matrix":confusion_mat,
         "build_time":formatted_build_time
    }

    return results
    
@app.post("/bagging")
async def bagging(
    estimators: int = Form(10, description="Number of base estimators in the ensemble"),
    base_estimator: str = Form('DecisionTree', description="Base estimator for bagging"),
    target_column: str = Form(...),
    independent_variables: List[str] = Form(...),
    train_percent: float = Form(...), 
    test_percent: float = Form(...), 
    validation_percent: float = Form(...)
):
    start_time=datetime.now()
    global table_name
    if table_name==' ':
        raise HTTPException(status_code=400, detail="First Upload source csv file")
      
    print(table_name)
    
    
    independent_variables_split = [f'"{col.strip()}"' for col in independent_variables[0].split(',')]
    independent_variables_joined = ', '.join(independent_variables_split)
    query = f"SELECT \"{target_column}\", {independent_variables_joined} FROM \"{table_name}\""

    data = pd.read_sql_query(query, conn)

    if target_column not in data.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data.")

    if data.isnull().values.any():
        raise HTTPException(status_code=400, detail="Data contains null values.")
    
    string_columns = data.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        raise HTTPException(status_code=400, detail="Data contains string values.")


    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split the dataset into training and testing sets
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size= train_percent, random_state=42)
    validation_size = validation_percent / (test_percent + validation_percent)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_size, random_state=42)
   
    if base_estimator == 'DecisionTree':
        base = DecisionTreeClassifier()
    elif base_estimator == 'KNeighbors':
        base = KNeighborsClassifier()
    elif base_estimator == 'SVM':
        base = SVC()
    elif base_estimator == 'LogisticRegression':
        base = LogisticRegression()
    elif base_estimator == 'NaiveBayes':
        base = GaussianNB()
    else:
        raise HTTPException(status_code=400, detail="Invalid base estimator specified")

    clf = BaggingClassifier(estimator=base, n_estimators=estimators)
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test) 

    unique_classes = len(np.unique(y_test))  
    if unique_classes == 2: 
        avg = 'binary'
    else:
        avg = 'macro'

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=avg)
    recall = recall_score(y_test, y_pred, average=avg) 
    f1 = f1_score(y_test, y_pred, average=avg)
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    recall_micro = recall_score(y_test, y_pred, average='micro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    logloss = log_loss(y_test, y_pred_proba) 
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    confusion_mat = confusion_matrix(y_test, y_pred).tolist()
    specificity = tn / (tn + fp)  
    sensitivity = recall  
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    balanced_accuracy = (specificity + sensitivity) / 2
    average_precision = average_precision_score(y_test, y_pred_proba[:, 1])

    end_time=datetime.now()
    build_time=end_time-start_time
    formatted_build_time=format_time(build_time)

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall (sensitivity)": sensitivity,
        "specificity": specificity,
        "f1_score": f1,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "recall_micro": recall_micro,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "log_loss": logloss,
        "auc": auc,
        "balanced_accuracy":balanced_accuracy,
        "average_precision":average_precision,
        "confusion_matrix":confusion_mat,
         "build_time":formatted_build_time
    }

    return results


@app.post("/boosting")
async def boosting(
    estimators: int = Form(50, description="The maximum number of estimators at which boosting is terminated."),
    boosting_algorithm: str = Form('AdaBoost', description="Boosting algorithm (AdaBoost, GradientBoosting, HistGradientBoosting)"),
    target_column: str = Form(...),
    independent_variables: List[str] = Form(...),
    train_percent: float = Form(...), 
    test_percent: float = Form(...), 
    validation_percent: float = Form(...)
):
    start_time=datetime.now()
    global table_name
    if table_name==' ':
        raise HTTPException(status_code=400, detail="First Upload source csv file")
      
    print(table_name)
    
    
    independent_variables_split = [f'"{col.strip()}"' for col in independent_variables[0].split(',')]
    independent_variables_joined = ', '.join(independent_variables_split)
    query = f"SELECT \"{target_column}\", {independent_variables_joined} FROM \"{table_name}\""

    data = pd.read_sql_query(query, conn)

    if target_column not in data.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data.")

    if data.isnull().values.any():
        raise HTTPException(status_code=400, detail="Data contains null values.")
    
    string_columns = data.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        raise HTTPException(status_code=400, detail="Data contains string values.")

    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split the dataset into training and testing sets
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size= train_percent, random_state=42)
    validation_size = validation_percent / (test_percent + validation_percent)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_size, random_state=42)
    
    # Choose the boosting algorithm based on user input
    if boosting_algorithm == 'AdaBoost':
        base_estimator = DecisionTreeClassifier()
        clf = AdaBoostClassifier(estimator=base_estimator, n_estimators=estimators)
    elif boosting_algorithm == 'GradientBoosting':
        clf = GradientBoostingClassifier(n_estimators=estimators)
    elif boosting_algorithm == 'HistGradientBoosting':
        clf = HistGradientBoostingClassifier(max_iter=estimators)
    elif boosting_algorithm == 'XGBoost':
        clf = xgb.XGBClassifier(n_estimators=estimators)
    else:
        raise HTTPException(status_code=400, detail="Invalid boosting algorithm specified")

    # Train the boosting classifier
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    unique_classes = len(np.unique(y_test))
    if unique_classes == 2: 
        avg = 'binary'
    else: 
        avg = 'macro'

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=avg)
    recall = recall_score(y_test, y_pred, average=avg) 
    f1 = f1_score(y_test, y_pred, average=avg)
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    recall_micro = recall_score(y_test, y_pred, average='micro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    logloss = log_loss(y_test, y_pred_proba)  
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    confusion_mat = confusion_matrix(y_test, y_pred).tolist()
    specificity = tn / (tn + fp) 
    sensitivity = recall  
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    sensitivity = recall
    balanced_accuracy = (specificity + sensitivity) / 2
    average_precision = average_precision_score(y_test, y_pred_proba[:, 1])

    end_time=datetime.now()
    build_time=end_time-start_time
    formatted_build_time=format_time(build_time)

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall (sensitivity)": sensitivity,
        "specificity": specificity,
        "f1_score": f1,
         "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "recall_micro": recall_micro,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "log_loss": logloss,
        "balanced_accuracy":balanced_accuracy,
        "average_precision":average_precision,
        "auc": auc,
        "confusion_matrix":confusion_mat,
         "build_time":formatted_build_time
    }

    return results

@app.post("/naive_bayes")
async def naive_bayes_classification(target_column: str = Form(...),
                                     train_percent: float = Form(...), 
                           test_percent: float = Form(...), 
                           validation_percent: float = Form(...),
                           independent_variables: List[str] = Form(...)):
    start_time=datetime.now()
    global table_name
    if table_name==' ':
        raise HTTPException(status_code=400, detail="First Upload source csv file")
      
    print(table_name)
        # query = f"SELECT * FROM {table_name}"
    independent_variables_split = [f'"{col.strip()}"' for col in independent_variables[0].split(',')]
    independent_variables_joined = ', '.join(independent_variables_split)
    query = f"SELECT \"{target_column}\", {independent_variables_joined} FROM \"{table_name}\""

    data = pd.read_sql_query(query, conn)

    if target_column not in data.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data.")
    
    if data.isnull().values.any():
        raise HTTPException(status_code=400, detail="Data contains null values.")
    
    string_columns = data.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        raise HTTPException(status_code=400, detail="Data contains string values.")

    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split the dataset into training and testing sets
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size= train_percent, random_state=42)
    validation_size = validation_percent / (test_percent + validation_percent)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_size, random_state=42)
    
    # Train the Naive Bayes classifier
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)

    unique_classes = len(np.unique(y_test)) 
    if unique_classes == 2: 
        avg = 'binary'
    else: 
        avg = 'macro'

    y_pred_proba = clf.predict_proba(X_test) 
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=avg)
    recall = recall_score(y_test, y_pred, average=avg)  
    f1 = f1_score(y_test, y_pred, average=avg)
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    recall_micro = recall_score(y_test, y_pred, average='micro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    logloss = log_loss(y_test, y_pred_proba)  
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    confusion_mat = confusion_matrix(y_test, y_pred).tolist()
    specificity = tn / (tn + fp) 
    sensitivity = recall  
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    sensitivity = recall
    balanced_accuracy = (specificity + sensitivity) / 2
    average_precision = average_precision_score(y_test, y_pred_proba[:, 1])
    end_time=datetime.now()
    build_time=end_time-start_time
    formatted_build_time=format_time(build_time)

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall (sensitivity)": sensitivity,
        "specificity": specificity,
        "f1_score": f1,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "recall_micro": recall_micro,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "log_loss": logloss,
        "auc": auc,
        "balanced_accuracy":balanced_accuracy,
         "average_precision":average_precision,
        "confusion_matrix":confusion_mat,
         "build_time":formatted_build_time
    }

    return results



@app.post("/knn")
async def knn(
        neighbors:int= Form(...),target_column: str = Form(...),
        train_percent: float = Form(...), 
                           test_percent: float = Form(...), 
                           validation_percent: float = Form(...),
                           independent_variables: List[str] = Form(...)
):
    start_time=datetime.now()
    global table_name
    if table_name==' ':
        raise HTTPException(status_code=400, detail="First Upload source csv file")
      
    print(table_name)
    # query = f"SELECT * FROM {table_name}"
    
    independent_variables_split = [f'"{col.strip()}"' for col in independent_variables[0].split(',')]
    independent_variables_joined = ', '.join(independent_variables_split)
    query = f"SELECT \"{target_column}\", {independent_variables_joined} FROM \"{table_name}\""

    data = pd.read_sql_query(query, conn)
    if target_column not in data.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data.")
    
    if data.isnull().values.any():
        raise HTTPException(status_code=400, detail="Data contains null values.")
    
    string_columns = data.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        raise HTTPException(status_code=400, detail="Data contains string values.")


    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split the dataset into training and testing sets
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size= train_percent, random_state=42)
    validation_size = validation_percent / (test_percent + validation_percent)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_size, random_state=42)
    
    # Train the Naive Bayes classifier
    clf = KNeighborsClassifier(n_neighbors=neighbors)
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    unique_classes = len(np.unique(y_test))  
    if unique_classes == 2:  
        avg = 'binary'
    else: 
        avg = 'macro'  
    y_pred_proba = clf.predict_proba(X_test)  
   
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=avg)
    recall = recall_score(y_test, y_pred, average=avg) 
    f1 = f1_score(y_test, y_pred, average=avg)
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    recall_micro = recall_score(y_test, y_pred, average='micro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    logloss = log_loss(y_test, y_pred_proba)  
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    confusion_mat = confusion_matrix(y_test, y_pred).tolist()
    specificity = tn / (tn + fp)  
    sensitivity = recall  
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    sensitivity = recall
    balanced_accuracy = (specificity + sensitivity) / 2
    average_precision = average_precision_score(y_test, y_pred_proba[:, 1])
    
    end_time=datetime.now()
    build_time=end_time-start_time
    formatted_build_time=format_time(build_time)

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall (sensitivity)": sensitivity,
        "specificity": specificity, 
        "f1_score": f1,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "recall_micro": recall_micro,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "log_loss": logloss,
        "auc": auc,
        "balanced_accuracy":balanced_accuracy,
        "average_precision":average_precision,
        "confusion_matrix":confusion_mat,
         "build_time":formatted_build_time
    }

    return results


@app.post("/svm-classification")
async def svm_classification(
    kernel: str = Form("rbf", description="Kernel function for SVM (e.g., 'linear', 'rbf', 'poly')"),
    C: float = Form(1.0, description="Regularization parameter"),
    gamma: str = Form("scale", description="Kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels"),
    target_column: str = Form(...),
    train_percent: float = Form(...), 
    test_percent: float = Form(...), 
    validation_percent: float = Form(...),
    independent_variables: List[str] = Form(...)
):
    start_time=datetime.now()
    global table_name
    if table_name==' ':
        raise HTTPException(status_code=400, detail="First Upload source csv file")
      
    print(table_name)

    
    independent_variables_split = [f'"{col.strip()}"' for col in independent_variables[0].split(',')]
    independent_variables_joined = ', '.join(independent_variables_split)
    query = f"SELECT \"{target_column}\", {independent_variables_joined} FROM \"{table_name}\""

    data = pd.read_sql_query(query, conn)

    if target_column not in data.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data.")

    if data.isnull().values.any():
        raise HTTPException(status_code=400, detail="Data contains null values.")
    
    string_columns = data.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        raise HTTPException(status_code=400, detail="Data contains string values.")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size= train_percent, random_state=42)
    validation_size = validation_percent / (test_percent + validation_percent)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_size, random_state=42)

    # Initialize SVM classifier
    clf = SVC(kernel=kernel, C=C, gamma=gamma)

    # Fit the classifier to the training data
    clf.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = clf.predict(X_test)

    unique_classes = len(np.unique(y_test))
    if unique_classes == 2:  
        avg = 'binary'
    else:  
        avg = 'macro'  

    y_pred_proba = clf.decision_function(X_test)  
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=avg)
    recall = recall_score(y_test, y_pred, average=avg)  
    f1 = f1_score(y_test, y_pred, average=avg)
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    recall_micro = recall_score(y_test, y_pred, average='micro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    logloss = log_loss(y_test, y_pred_proba)  # Log Loss
    confusion_mat = confusion_matrix(y_test, y_pred).tolist()
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    auc = roc_auc_score(y_test, y_pred_proba)
    sensitivity = recall
    balanced_accuracy = (specificity + sensitivity) / 2
    average_precision = average_precision_score(y_test, y_pred_proba)
    
    end_time=datetime.now()
    build_time=end_time-start_time
    formatted_build_time=format_time(build_time)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall (sensitivity)": recall,
        "specificity": specificity,
        "f1_score": f1,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "recall_micro": recall_micro,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "log_loss": logloss,
        "auc": auc,
        "balanced_accuracy":balanced_accuracy,
        "average_precision":average_precision,
        "confusion_matrix": confusion_mat,
        "build_time":formatted_build_time
    }
    
@app.post("/linear")
async def predict(independent_variables: List[str] = Form(...),
    target_column: str = Form(...),
                  train_percent: float = Form(...), 
                    test_percent: float = Form(...), 
                    validation_percent: float = Form(...)):
    try:
        start_time=datetime.now()
        global table_name
        if table_name==' ':
            raise HTTPException(status_code=400, detail="First Upload source csv file")
        
        print(table_name)
            # query = f"SELECT * FROM {table_name}"
        
        independent_variables_split = [f'"{col.strip()}"' for col in independent_variables[0].split(',')]
        independent_variables_joined = ', '.join(independent_variables_split)
        query = f"SELECT \"{target_column}\", {independent_variables_joined} FROM \"{table_name}\""
        data = pd.read_sql_query(query, conn)

        print("data",data)
        if target_column not in data.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data.")
        
        if data.isnull().values.any():
            raise HTTPException(status_code=400, detail="Data contains null values.")
        
        string_columns = data.select_dtypes(include=['object']).columns
        if len(string_columns) > 0:
            raise HTTPException(status_code=400, detail="Data contains string values.")

        X = data.drop(columns=[target_column])
        y = data[target_column]
            
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size= train_percent, random_state=42)
        validation_size = validation_percent / (test_percent + validation_percent)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_size, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        # Calculate adjusted R-squared
        n = len(y_test)
        p = X_test.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        mean_error = np.mean(y_test - y_pred)
        rmsle=np.log(rmse)
        evs = explained_variance_score(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)

        end_time=datetime.now()
        build_time=end_time-start_time
        formatted_build_time=format_time(build_time)


        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,   
            "r2": r2,
            "adj_r2": adj_r2,
            "mean_error": mean_error,
            "rmsle": rmsle,
            "explained_variance": evs,
            "median_absolute_error": medae,
            "build_time":formatted_build_time
        }
    except Exception as e:
         raise HTTPException(status_code=400, detail=str(e))


@app.post("/polynomial")
async def polynomial_regression(degree: int = Form(2),target_column: str = Form(...),
                                independent_variables: List[str] = Form(...),
                                 train_percent: float = Form(...), 
                           test_percent: float = Form(...), 
                           validation_percent: float = Form(...)):
    try:
        start_time=datetime.now()

        global table_name
        if table_name==' ':
            raise HTTPException(status_code=400, detail="First Upload source csv file")
        
        print(table_name)
        # query = f"SELECT * FROM {table_name}"
            
        
        independent_variables_split = [f'"{col.strip()}"' for col in independent_variables[0].split(',')]
        independent_variables_joined = ', '.join(independent_variables_split)
        query = f"SELECT \"{target_column}\", {independent_variables_joined} FROM \"{table_name}\""

        data = pd.read_sql_query(query, conn)

        if target_column not in data.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data.")

        # Check for null values
        if data.isnull().values.any():
            raise HTTPException(status_code=400, detail="Data contains null values.")

        # Check for string values
        string_columns = data.select_dtypes(include=['object']).columns
        if len(string_columns) > 0:
            raise HTTPException(status_code=400, detail="Data contains string values.")
        
        
        # Create polynomial features
        X_poly = PolynomialFeatures(degree=degree).fit_transform(data.drop(columns=[target_column]))

        y = data[target_column]

        X_train, X_temp, y_train, y_temp = train_test_split(X_poly, y, train_size= train_percent, random_state=42)
        validation_size = validation_percent / (test_percent + validation_percent)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_size, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        n = len(y_test)
        p = X_test.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        # Calculate MAPE - Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        mean_error = np.mean(y_test - y_pred)
        # Calculate RMSLE - Root Mean Squared Log Error
        rmsle=np.log(rmse)
        evs = explained_variance_score(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)

        end_time=datetime.now()
        build_time=end_time-start_time
        formatted_build_time=format_time(build_time)

        return {
            "mse": mse, 
            "mae": mae, 
            "rmse": rmse, 
            "r2": r2, 
            "adj_r2": adj_r2,
            "mean_error": mean_error,
            "rmsle": rmsle,
            "explained_variance": evs,
            "median_absolute_error": medae,
            "build_time":formatted_build_time
        }
    except Exception as e:
        raise HTTPException(status_code=400,detail=str(e)) 

@app.post("/logistic")
async def predict_logistic(target_column: str = Form(...),
                           independent_variables: List[str] = Form(...),
                           train_percent: float = Form(...), 
                           test_percent: float = Form(...), 
                           validation_percent: float = Form(...)):
    start_time=datetime.now()

    global table_name
    if table_name==' ':
        raise HTTPException(status_code=400, detail="First Upload source csv file")
      
    print(table_name)
    # query = f"SELECT * FROM {table_name}"
    # query = f"SELECT {target_column}, {', '.join(independent_variables)} FROM {table_name}"
    
    independent_variables_split = [f'"{col.strip()}"' for col in independent_variables[0].split(',')]
    independent_variables_joined = ', '.join(independent_variables_split)
    query = f"SELECT \"{target_column}\", {independent_variables_joined} FROM \"{table_name}\""

    data = pd.read_sql_query(query, conn)
    
    print(data)
     
    if target_column not in data.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data.")

    if data.isnull().values.any():
        raise HTTPException(status_code=400, detail="Data contains null values.")
    
    string_columns = data.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        raise HTTPException(status_code=400, detail="Data contains string values.")

   

    # Split data into features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    print("y",y)
    # Split data into train and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size= train_percent, random_state=42)
    validation_size = validation_percent / (test_percent + validation_percent)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_size, random_state=42)


    model = LogisticRegression()

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    rmsle=np.log(rmse)
    msle = mean_squared_log_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)

    end_time=datetime.now()
    build_time=end_time-start_time
    formatted_build_time=format_time(build_time)
  
    results = {
        
        "mse": mse, 
        "mae": mae, 
        "rmse": rmse, 
        "rmsle":rmsle,
        "r2": r2,
        # "mean_squared_log_error": msle,
        "explained_variance": evs,
        "median_absolute_error": medae,
        "built_time":formatted_build_time
    }
    return results

    
@app.post("/decision-tree-regression")
async def decision_tree_regression(
    criterion: str = Form("squared_error", description="The function to measure the quality of a split"),
    max_depth: int = Form(None, description="The maximum depth of the tree"),
    min_samples_split: int = Form(2, description="The minimum number of samples required to split an internal node"),
    min_samples_leaf: int = Form(1, description="The minimum number of samples required to be at a leaf node"),
    max_leaf_nodes: int = Form(None, description="Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity."),
    target_column: str = Form(...),
    train_percent: float = Form(...), 
    test_percent: float = Form(...), 
    validation_percent: float = Form(...),
    independent_variables: List[str] = Form(...)
):
    start_time = datetime.now()
    
    global table_name
    if table_name==' ':
        raise HTTPException(status_code=400, detail="First Upload source csv file")
      
    print(table_name)
    # query = f"SELECT * FROM {table_name}"

# query = f"SELECT {target_column}, {', '.join(independent_variables)} FROM {table_name}"
    
    independent_variables_split = [f'"{col.strip()}"' for col in independent_variables[0].split(',')]
    independent_variables_joined = ', '.join(independent_variables_split)
    query = f"SELECT \"{target_column}\", {independent_variables_joined} FROM \"{table_name}\""

    data = pd.read_sql_query(query, conn)
    print("data",data)
    if target_column not in data.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data.")

    if data.isnull().values.any():
        raise HTTPException(status_code=400, detail="Data contains null values.")
    
    string_columns = data.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        raise HTTPException(status_code=400, detail="Data contains string values.")
    
    X = data.drop(columns=[target_column])
    y = data[target_column]

    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size= train_percent, random_state=42)
    validation_size = validation_percent / (test_percent + validation_percent)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_size, random_state=42)

    # Initialize DecisionTreeRegressor with specified parameters
    regressor = DecisionTreeRegressor(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=max_leaf_nodes
    )

    # Fit the model
    regressor.fit(X_train, y_train)

    # Make predictions
    y_pred = regressor.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse) 
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    mean_error = np.mean(y_test - y_pred)
    rmsle=np.log(rmse)
    msle = mean_squared_log_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)

    end_time=datetime.now()
    build_time=end_time-start_time
    formatted_build_time=format_time(build_time)

    results = {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        # "mape": mape,
        "rmsle":rmsle,
        # "mean_squared_log_error": msle,
        "explained_variance": evs,
        "median_absolute_error": medae,
        "mean_error": mean_error,
        "built_time":formatted_build_time
    }

    return results

@app.post("/random-forest-regressior")
async def random_forest_regression(
    estimators:int=Form(...,description="number of decision tree"),
    criterion:str=Form(...,description="The function to measure the quality of a split"),
    target_column: str = Form(...),
    train_percent: float = Form(...), 
    test_percent: float = Form(...), 
    validation_percent: float = Form(...),
    independent_variables: List[str] = Form(...)
):
    try:
        start_time=datetime.now()

        global table_name
        if table_name==' ':
            raise HTTPException(status_code=400, detail="First Upload source csv file")
        
        print(table_name)
        # query = f"SELECT * FROM {table_name}"

        # query = f"SELECT {target_column}, {', '.join(independent_variables)} FROM {table_name}"
       
            
        
        independent_variables_split = [f'"{col.strip()}"' for col in independent_variables[0].split(',')]
        independent_variables_joined = ', '.join(independent_variables_split)
        query = f"SELECT \"{target_column}\", {independent_variables_joined} FROM \"{table_name}\""

        data = pd.read_sql_query(query, conn)

        if target_column not in data.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data.")

        if data.isnull().values.any():
            raise HTTPException(status_code=400, detail="Data contains null values.")
        
        string_columns = data.select_dtypes(include=['object']).columns
        if len(string_columns) > 0:
            raise HTTPException(status_code=400, detail="Data contains string values.")


        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Split the dataset into training and testing sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size= train_percent, random_state=42)
        validation_size = validation_percent / (test_percent + validation_percent)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_size, random_state=42)

        
        clf=RandomForestRegressor(n_estimators=estimators,criterion=criterion)
        
        #fit the model
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_test)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)  
        r2 = r2_score(y_test, y_pred)
        mean_error = np.mean(y_test - y_pred)
        if rmse > 0:
            rmsle = np.log(rmse)
        else:
            rmsle = None
        evs = explained_variance_score(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)

        end_time=datetime.now()
        build_time=end_time-start_time
        formatted_build_time=format_time(build_time)

        results = {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "rmsle":rmsle,
            "explained_variance": evs,
            "median_absolute_error": medae,
            "mean_error": mean_error,
            "built_time":formatted_build_time
        }

        return results
    except Exception as e:
        raise HTTPException(status_code=400,detail=str(e))  




@app.post("/svm-regression")
async def svm_regression(
    kernel: str = Form("rbf", description="Specifies the kernel type to be used in the algorithm ('linear', 'poly', 'rbf', 'sigmoid')"),
    C: float = Form(1.0, description="Regularization parameter"),
    gamma: float = Form("scale", description="Kernel coefficient for 'rbf', 'poly', and 'sigmoid'. 'scale' uses 1 / (n_features * X.var()) as value."),
    degree: int = Form(3, description="Degree of the polynomial kernel function ('poly' kernel only)"),
    target_column: str = Form(...),
    train_percent: float = Form(...), 
    test_percent: float = Form(...), 
    validation_percent: float = Form(...),
    independent_variables: List[str] = Form(...)
):
    start_time=datetime.now()
    global table_name
    if table_name==' ':
        raise HTTPException(status_code=400, detail="First Upload source csv file")
      
    print(table_name)
    # query = f"SELECT * FROM {table_name}"
    # query = f"SELECT {target_column}, {', '.join(independent_variables)} FROM {table_name}"

    
    independent_variables_split = [f'"{col.strip()}"' for col in independent_variables[0].split(',')]
    independent_variables_joined = ', '.join(independent_variables_split)
    query = f"SELECT \"{target_column}\", {independent_variables_joined} FROM \"{table_name}\""

    data = pd.read_sql_query(query, conn)

    if target_column not in data.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data.")

    if data.isnull().values.any():
        raise HTTPException(status_code=400, detail="Data contains null values.")
    
    string_columns = data.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        raise HTTPException(status_code=400, detail="Data contains string values.")


    X = data.drop(columns=[target_column])
    y = data[target_column]

    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size= train_percent, random_state=42)
    validation_size = validation_percent / (test_percent + validation_percent)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_size, random_state=42)

 
    # Initialize SVR with specified parameters
    svr = SVR(kernel=kernel, C=C, gamma=gamma, degree=degree)

    # Fit the model
    svr.fit(X_train, y_train)

    # Make predictions
    y_pred = svr.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse) 
    r2 = r2_score(y_test, y_pred)
    rmsle=np.log(rmse)
    msle = mean_squared_log_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)

    end_time=datetime.now()
    build_time=end_time-start_time
    formatted_build_time=format_time(build_time)

    results = {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        # "mean_squared_log_error": msle,
        "explained_variance": evs,
        "median_absolute_error": medae,
        "rmsle":rmsle,
        "build_time":formatted_build_time
    }

    return results

def check_pca_analysis(data: pd.DataFrame) -> bool:
 
    pca_columns_present = any(col.startswith('PC') for col in data.columns)
 
    return pca_columns_present


def check_null_values(data: pd.DataFrame) -> bool:
    return data.isnull().values.any()


def check_duplicates(data: pd.DataFrame) -> bool:
    return data.duplicated().any()

def check_multicollinearity(data: pd.DataFrame, threshold: float = 0.9) -> bool:
    correlation_matrix = data.corr()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    return (upper_triangle.abs() > threshold).any().any()

def check_string_values(data: pd.DataFrame) -> bool:
    # Check if any column has string values
    has_strings = any(data[col].dtype == 'object' for col in data.columns)
    return not has_strings

@app.post("/check_data")
async def check_data():
    try:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)

        # Check if string values are present
        if check_string_values(df):
            # Execute the existing checks
            null_values =  check_null_values(df)
            duplicates =  check_duplicates(df)
            multicollinearity =  check_multicollinearity(df)
            pca_analysis_done = check_pca_analysis(df)

            results = {
                "null_values": bool(null_values),
                "duplicates": bool(duplicates),
                "multicollinearity": bool(multicollinearity),
                "pca_analysis_done": pca_analysis_done
            }
            return results
        else:
            return {"message": "String values are present in the DataFrame. Cannot execute checks."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))