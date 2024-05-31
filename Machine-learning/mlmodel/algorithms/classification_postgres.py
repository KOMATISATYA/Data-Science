from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score,roc_auc_score,log_loss,average_precision_score
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from io import BytesIO
import xgboost as xgb
import mysql.connector
from datetime import datetime
import psycopg2


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

@app.post("/decision-tree-classification")
async def decision_tree_classification(
    criterion: str = Form("gini", description="The function to measure the quality of a split"),
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
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
    

@app.post("/random-forest-classification")
async def randomForest(
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
    base_estimator: str = Form('DecisionTree', description="Base estimator for bagging")
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
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
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
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
async def naive_bayes_classification():
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
        neighbors:int= Form(...)
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
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
    


 