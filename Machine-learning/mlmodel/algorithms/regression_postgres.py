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
from io import BytesIO
import psycopg2
from datetime import datetime
from sklearn.svm import SVR


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


app = FastAPI()

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
target_column=''

@app.post("/source")
async def source(file: UploadFile = File(...),
                 target: str = Form(...)):
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
        file_csv = file_csv[0].lower() + file_csv[1:]
        global table_name
        table_name=file_csv
        save_file_processed(df,file_csv)
        global target_column
        target_column=target
        return "success"
    except HTTPException as http_err:
        print("httpexception") 
        raise http_err
    except Exception as e :
        print("eeeexception") 
        return e  


@app.post("/linear")
async def predict():
    start_time=datetime.now()
    # Read the file content from the UploadFile object
    global table_name
    print(table_name)
    query = f"SELECT * FROM {table_name}"
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


@app.post("/polynomial")
async def polynomial_regression(degree: int = Form(2)):
    # Read the file content from the UploadFile object
    start_time=datetime.now()

    global table_name
    print(table_name)
    query = f"SELECT * FROM {table_name}"
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
async def predict_logistic():
    # Read the file content from the UploadFile object
    start_time=datetime.now()

    global table_name
    print(table_name)
    query = f"SELECT * FROM {table_name}"
    data = pd.read_sql_query(query, conn)

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
    criterion: str = Form("squared_error", description="The function to measure the quality of a split"),
    max_depth: int = Form(None, description="The maximum depth of the tree"),
    min_samples_split: int = Form(2, description="The minimum number of samples required to split an internal node"),
    min_samples_leaf: int = Form(1, description="The minimum number of samples required to be at a leaf node"),
    max_leaf_nodes: int = Form(None, description="Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity.")
):
    start_time = datetime.now()
    
    global table_name
    print(table_name)
    query = f"SELECT * FROM {table_name}"
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
    estimators:int=Form(...,description="number of decision tree"),
    criterion:str=Form(...,description="The function to measure the quality of a split")):

    start_time=datetime.now()

    global table_name
    print(table_name)
    query = f"SELECT * FROM {table_name}"
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
    kernel: str = Form("rbf", description="Specifies the kernel type to be used in the algorithm ('linear', 'poly', 'rbf', 'sigmoid')"),
    C: float = Form(1.0, description="Regularization parameter"),
    gamma: float = Form("scale", description="Kernel coefficient for 'rbf', 'poly', and 'sigmoid'. 'scale' uses 1 / (n_features * X.var()) as value."),
    degree: int = Form(3, description="Degree of the polynomial kernel function ('poly' kernel only)")
):
    start_time=datetime.now()
    global table_name
    print(table_name)
    query = f"SELECT * FROM {table_name}"
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


