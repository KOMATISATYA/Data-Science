from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_squared_log_error, explained_variance_score, median_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report,r2_score
import numpy as np
import io
import mysql.connector
import statsmodels.api as sm
from datetime import datetime
from sklearn.svm import SVR


conn = mysql.connector.connect(
    user = "root",
    password = "root",
    host = "localhost",
    port = "3306"
)


conn.autocommit = True
cursor = conn.cursor()
database_name = 'mlregression'
create_query = f"CREATE DATABASE IF NOT EXISTS {database_name}"
cursor.execute(create_query)
query2=f"use {database_name}"
cursor.execute(query2)
conn.commit()
result = cursor.fetchall()
conn.commit() 


app = FastAPI()

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
        # Add column definition to create table query
        create_table_query += f"`{column}` {mysql_type}, "
    create_table_query = create_table_query.rstrip(', ') + ")"
    cursor.execute(create_table_query)
    # Insert data into table
    for _, row in df.iterrows():
        # Replace null or missing values with None
        row = row.where(pd.notnull(row), None)

        # Prepare column names and values for insertion
        columns = ', '.join([f"`{key}`" for key in row.keys()])
        placeholders = ', '.join(['%s'] * len(row))
        values = tuple(map(lambda x: x.item() if isinstance(x, np.generic) else x, row.values))
        insert_query = f"INSERT INTO `{table_name}` ({columns}) VALUES ({placeholders})"
        cursor.execute(insert_query, values)
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


@app.post("/linear")
async def predict(file: UploadFile = File(...), target_column: str = Form(...)):
    start_time=datetime.now()
    # Read the file content from the UploadFile object
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

    if target_column not in data.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data.")
    
    if data.isnull().values.any():
        raise HTTPException(status_code=400, detail="Data contains null values.")
    
    string_columns = data.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        raise HTTPException(status_code=400, detail="Data contains string values.")

    csv_filee = file.filename.split("/")
    file_csv=csv_filee[-1].replace(".csv","_table")
    save_file_processed(data, file_csv)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    # Calculate MAPE - Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    mean_error = np.mean(y_test - y_pred)
    rmsle=np.log(rmse)
    # msle = mean_squared_log_error(y_test, y_pred)
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
        "mape": mape,
        "mean_error": mean_error,
        "rmsle": rmsle,
        # "mean_squared_log_error": msle,
        "explained_variance": evs,
        "median_absolute_error": medae,
        "build_time":formatted_build_time
    }


@app.post("/polynomial")
async def polynomial_regression(file: UploadFile = File(...), target_column: str = Form(...), degree: int = Form(2)):
    # Read the file content from the UploadFile object
    start_time=datetime.now()
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

    if target_column not in data.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data.")

    # Check for null values
    if data.isnull().values.any():
        raise HTTPException(status_code=400, detail="Data contains null values.")

    # Check for string values
    string_columns = data.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        raise HTTPException(status_code=400, detail="Data contains string values.")
     
    
    csv_filee = file.filename.split("/")

    file_csv=csv_filee[-1].replace(".csv","_table")
 
    save_file_processed(data, file_csv)
    
    # Create polynomial features
    X_poly = PolynomialFeatures(degree=degree).fit_transform(data.drop(columns=[target_column]))

    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

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
    msle = mean_squared_log_error(y_test, y_pred)
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
        "mape": mape,
        "mean_error": mean_error,
        "rmsle": rmsle,
        "mean_squared_log_error": msle,
        "explained_variance": evs,
        "median_absolute_error": medae,
        "build_time":formatted_build_time
    }

@app.post("/logistic")
async def predict_logistic(file: UploadFile = File(...), target_column: str = Form(...)):
    # Read the file content from the UploadFile object
    start_time=datetime.now()
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

    if target_column not in data.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data.")

    if data.isnull().values.any():
        raise HTTPException(status_code=400, detail="Data contains null values.")
    
    string_columns = data.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        raise HTTPException(status_code=400, detail="Data contains string values.")

    # Save data to database
    table_name = file.filename.split("/")[-1].replace(".csv", "_table")
    save_file_processed(data, table_name)

    # Split data into features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)
    class_report = classification_report(y_test, y_pred, output_dict=True)
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
        "classification_report": class_report,
        "mse": mse, 
        "mae": mae, 
        "rmse": rmse, 
        "rmsle":rmsle,
        "r2": r2,
        "mean_squared_log_error": msle,
        "explained_variance": evs,
        "median_absolute_error": medae,
        "built_time":formatted_build_time
    }
    return results
    
@app.post("/decision-tree-regression")
async def decision_tree_regression(
    file: UploadFile = File(...),
    target_column: str = Form(..., description="Name of the target column"),
    criterion: str = Form("squared_error", description="The function to measure the quality of a split"),
    max_depth: int = Form(None, description="The maximum depth of the tree"),
    min_samples_split: int = Form(2, description="The minimum number of samples required to split an internal node"),
    min_samples_leaf: int = Form(1, description="The minimum number of samples required to be at a leaf node"),
    max_leaf_nodes: int = Form(None, description="Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity.")
):
    start_time = datetime.now()
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

    if target_column not in data.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data.")

    if data.isnull().values.any():
        raise HTTPException(status_code=400, detail="Data contains null values.")
    
    string_columns = data.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        raise HTTPException(status_code=400, detail="Data contains string values.")
    
    # Save data to database
    table_name = file.filename.split("/")[-1].replace(".csv", "_table")
    save_file_processed(data, table_name)


    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
        "mape": mape,
        "rmsle":rmsle,
        "mean_squared_log_error": msle,
        "explained_variance": evs,
        "median_absolute_error": medae,
        "mean_error": mean_error,
        "built_time":formatted_build_time
    }

    return results

@app.post("/random-forest-regressior")
async def random_forest_regression(
    file: UploadFile = File(...),
    target_column: str = Form(..., description="Name of the target column"),
    estimators:int=Form(...,description="number of decision tree"),
    criterion:str=Form(...,description="The function to measure the quality of a split")):

    start_time=datetime.now()
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

    if target_column not in data.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data.")

    if data.isnull().values.any():
        raise HTTPException(status_code=400, detail="Data contains null values.")
    
    string_columns = data.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        raise HTTPException(status_code=400, detail="Data contains string values.")

    # Save data to database
    table_name = file.filename.split("/")[-1].replace(".csv", "_table")
    save_file_processed(data, table_name)

    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
        "mape": mape,
        "rmsle":rmsle,
        "mean_squared_log_error": msle,
        "explained_variance": evs,
        "median_absolute_error": medae,
        "mean_error": mean_error,
        "built_time":formatted_build_time
    }

    return results



@app.post("/svm-regression")
async def svm_regression(
    file: UploadFile = File(...),
    target_column: str = Form(..., description="Name of the target column"),
    kernel: str = Form("rbf", description="Specifies the kernel type to be used in the algorithm ('linear', 'poly', 'rbf', 'sigmoid')"),
    C: float = Form(1.0, description="Regularization parameter"),
    gamma: float = Form("scale", description="Kernel coefficient for 'rbf', 'poly', and 'sigmoid'. 'scale' uses 1 / (n_features * X.var()) as value."),
    degree: int = Form(3, description="Degree of the polynomial kernel function ('poly' kernel only)")
):
    start_time=datetime.now()
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

    if target_column not in data.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data.")

    if data.isnull().values.any():
        raise HTTPException(status_code=400, detail="Data contains null values.")
    
    string_columns = data.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        raise HTTPException(status_code=400, detail="Data contains string values.")
    # Save data to database
    table_name = file.filename.split("/")[-1].replace(".csv", "_table")
    save_file_processed(data, table_name)

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 
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
        "mean_squared_log_error": msle,
        "explained_variance": evs,
        "median_absolute_error": medae,
        "rmsle":rmsle,
        "build_time":formatted_build_time
    }

    return results


