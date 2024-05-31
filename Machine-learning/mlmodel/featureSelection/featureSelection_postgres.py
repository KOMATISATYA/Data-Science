from fastapi import FastAPI,UploadFile,File,Form,HTTPException
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_classif,r_regression,mutual_info_classif,mutual_info_regression
import io
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.svm import SVR,SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
import psycopg2
from io import BytesIO
from sklearn.feature_selection import RFECV


app=FastAPI()

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


@app.post("/variance_filter")
async def variance_threshold_selector(
      threshold: float = Form(0.0)):
    try:
        global table_name
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
      threshold: float = Form(0.0)):
    try:
        global table_name
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
      method: str = Form("f_classif")):
    try:
        global table_name
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
        return RFECV(estimator=svc)
    else:
        raise HTTPException(status_code=400, detail="Invalid estimator method")

@app.post("/recursive_feature")
async def recursive_feature_elimination(
      n: int = Form(...),
      method: str = Form(...)):
    try:
        global table_name
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
    alpha: float = Form(...)
):
    try:
        global table_name
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
async def perform_pca():
    # Read the uploaded file
    

    global table_name
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

        # Save the combined and transformed data to the database table
    save_file_processed(transformed_data, processed_table_name)

    return {"transformed_features": X_pca.tolist()}