from fastapi import FastAPI, File, UploadFile, Form,HTTPException
import numpy as np
import pandas as pd
import io

app=FastAPI()

classification_algorithms = ['Decision Tree', 'Random Forest', 'SVM']
regression_algorithms = ['Linear Regression', 'Random Forest Regressor', 'XGBoost']
clustering_algorithms = ['kmeans','mean shift','Hierarchial','Divisive']



@app.post("/algorithm-selection")
async def appropiate_algorithm(
    file: UploadFile = File(...),
    target_column: str = Form(..., description="Name of the target column")
):
    
    content = await file.read()

    # Use io.BytesIO to convert the content to a file-like object
    file_like_object = io.BytesIO(content)
    data = pd.read_csv(file_like_object)  # Assuming the uploaded file is a CSV

    # Check if the target column exists in the dataset
    if target_column not in data.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data.")

    # Extract the target column from the dataset
    target_values = data[target_column]
    
    # if len( target_values.unique()) == 2:
        # Check if unique values are not numeric but represent class labels
    # if all(isinstance(val, str) for val in  target_values.unique()):
    #         return {"algorithm": classification_algorithms}
    # elif  target_values.dtype in [np.int64, np.float64]:
    #     return {"algorithm": regression_algorithms}
    # # Default to multiclass classification
    # return {"algorithm": clustering_algorithms}
     
    if len( target_values.unique()) == 2:
        return 'categorical'

    elif pd.api.types.is_numeric_dtype(target_values):
        return 'continuous'
    else:
        return 'none'


