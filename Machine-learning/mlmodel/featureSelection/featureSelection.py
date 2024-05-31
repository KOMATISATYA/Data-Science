from fastapi import FastAPI,UploadFile,File,Form,HTTPException
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_classif,r_regression,mutual_info_classif,mutual_info_regression
import io
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR,SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS


app=FastAPI()

@app.post("/variance_filter")
async def variance_threshold_selector(
    file: UploadFile = File(...),
      target_column: str = Form(...),
      threshold:float =Form(0.0)):
    print(threshold)
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
    
    string_columns = data.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        raise HTTPException(status_code=400, detail="Data contains string values. please do encoing step first")
    
    X = data.drop(columns=[target_column])
    y = data[target_column]


    selector = VarianceThreshold(threshold)
    selector.fit(X)
    return X.columns[selector.get_support(indices=True)].tolist()


@app.post("/correlation_selector")
async def correlation_selector( file: UploadFile = File(...),
      target_column: str = Form(...),
      threshold:float =Form(0.0)):
    
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
    
    string_columns = data.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        raise HTTPException(status_code=400, detail="Data contains string values. please do encoing step first")
    
    
    X = data.drop(columns=[target_column])
    y = data[target_column]


    cor = X.corrwith(y)
    return cor[abs(cor) > threshold].index.tolist()

@app.post("/univariate_selector")
async def univariate_selector(
      file: UploadFile = File(...),
      target_column: str = Form(...),
      k:int =Form(10) ,
      method:str=Form("f_classif")):
    
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

        raise HTTPException(status_code=400, detail="Data contains null values. Please perform missing values preprocessing step")
    
    string_columns = data.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        raise HTTPException(status_code=400, detail="Data contains string values. please do encoing step first")
    
    X = data.drop(columns=[target_column])
    y = data[target_column]

    
    if method == 'chi2':
        selector = SelectKBest(chi2, k=k)
    elif method =='fclassif':
        selector = SelectKBest(f_classif, k=k)
    elif method == 'mutual_info_classify':
        selector = SelectKBest(mutual_info_classif,k=k)
    elif method == 'r_regression':
        selector = SelectKBest(r_regression,k=k)
    elif method == 'mutual_info_regression':
        selector = SelectKBest(mutual_info_regression,k=k)
    selector.fit(X, y)
    return X.columns[selector.get_support(indices=True)].tolist()


def get_estimator(method):
    if method == "linear_regression":
        return LinearRegression()
    elif method == "random_forest_regressor":
        return RandomForestRegressor()
    elif method == "svr":
        return SVR()
    elif method == 'svc':
        return SVC()
    else:
        raise HTTPException(status_code=400, detail="Invalid estimator method")

@app.post("/recursive_feature")
async def recursive_feature_elemination(
      file: UploadFile = File(...),
      target_column: str = Form(...),
      n:int =Form(...) ,
      method:str=Form(...)
):
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

        raise HTTPException(status_code=400, detail="Data contains null values. Please perform missing values preprocessing step")
    
    string_columns = data.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        raise HTTPException(status_code=400, detail="Data contains string values. please do encoing step first")
    

    estimator = get_estimator(method)
    rfe = RFE(estimator=estimator, n_features_to_select=n)

    X = data.drop(columns=[target_column])
    y = data[target_column]

    rfe.fit(X, y)
    
    return X.columns[rfe.get_support(indices=True)].tolist()

@app.post("/lasso_feature_selection")
async def LassoFeatureSelection(
    file: UploadFile = File(...),
    target_column:str=Form(...),
    alpha: float = Form(...),
):
    
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
    
    if data.isnull().values.any():

        raise HTTPException(status_code=400, detail="Data contains null values. Please perform missing values preprocessing step")
    
    string_columns = data.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        raise HTTPException(status_code=400, detail="Data contains string values. please do encoing step first")
    
    
        # Split features and target
    X = data.drop(columns=[target_column])  # Assuming the target column is named "target"
    y = data[target_column]

       
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

       
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_scaled, y)

        # Extract selected features based on non-zero coefficients
    selected_features_indices = [i for i, coef in enumerate(lasso.coef_) if coef != 0]
    selected_features = X.columns[selected_features_indices].tolist()

    return {"selected_features": selected_features}


async def perform_pca(file: UploadFile,target_column):
    # Read the uploaded file
    
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
    string_columns = data.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        raise HTTPException(status_code=400, detail="Data contains string values. please do encoing step first")
    if data.isnull().values.any():

        raise HTTPException(status_code=400, detail="Data contains null values. Please perform missing values preprocessing step")
    
    X = data.drop(columns=[target_column])  # Assuming the target column is named "target"
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize PCA
    pca = PCA()

    # Fit PCA
    pca.fit(X_scaled)

    # Determine the number of components to retain
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    
    print("components",num_components)
    
    pca = PCA(n_components=num_components)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca

@app.post("/pca")
async def pca_endpoint(file: UploadFile = File(...), target_column:str=Form(...),):
    transformed_features = await perform_pca(file,target_column)
    return {"transformed_features": transformed_features.tolist()}
    
   
