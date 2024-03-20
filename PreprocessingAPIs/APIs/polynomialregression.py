# from fastapi import FastAPI, File, UploadFile, Form, HTTPException
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split
# import numpy as np
# import io
# import mysql.connector

# conn=mysql.connector.connect(
#     user = "root",
#     password = "root",
#     host = "localhost",
#     port = "3306"
# )

# conn.autocommit = True
# cursor = conn.cursor()
# database_name = 'mlmodels'
# create_query = f"CREATE DATABASE IF NOT EXISTS {database_name}"
# cursor.execute(create_query)
# query2=f"use {database_name}"
# cursor.execute(query2)
# conn.commit()
# result = cursor.fetchall()
# print(result)
# conn.commit()


# app=FastAPI()

# def save_file_processed(df, table_name):
#     print("....")
#     print(df)
#     create_table_query = f"CREATE TABLE IF NOT EXISTS `{table_name}` ("
#     for column, dtype in df.dtypes.items():
#         # Map pandas dtype to MySQL data type
#         mysql_type = "VARCHAR(255)"  # Default to VARCHAR(255) if type is not recognized
#         if dtype == 'int64':
#             mysql_type = 'BIGINT'
#         elif dtype == 'float64':
#             mysql_type = 'FLOAT'
#         elif dtype == 'object':
#             # Check if it's a string or datetime
#             if pd.api.types.is_string_dtype(df[column]):
#                 mysql_type = 'VARCHAR(255)'
#             elif pd.api.types.is_datetime64_any_dtype(df[column]):
#                 mysql_type = 'DATETIME'
#         # Add column definition to create table query
#         create_table_query += f"`{column}` {mysql_type}, "
#     create_table_query = create_table_query.rstrip(', ') + ")"
#     cursor.execute(create_table_query)
#     print("created..")
#     # Insert data into table
#     for _, row in df.iterrows():
#         # Replace null or missing values with None
#         row = row.where(pd.notnull(row), None)

#         # Prepare column names and values for insertion
#         columns = ', '.join([f"`{key}`" for key in row.keys()])
#         print("columns", columns)
#         placeholders = ', '.join(['%s'] * len(row))
#         print(placeholders)
#         values = tuple(map(lambda x: x.item() if isinstance(x, np.generic) else x, row.values))
#         print("values", values)

#         # Insert row into table
#         insert_query = f"INSERT INTO `{table_name}` ({columns}) VALUES ({placeholders})"
#         print("insert query")
#         cursor.execute(insert_query, values)
#         print("inserted..")
#     conn.commit()

# @app.post("/polynomial")
# async def polynomial_regression(file: UploadFile = File(...), target_column: str = Form(...), degree: int = Form(2)):
#     # Read the file content from the UploadFile object
#     content = await file.read()
#     print(content)
#     # Use io.BytesIO to convert the content to a file-like object
#     file_like_object = io.BytesIO(content)

#     # Read the data from the file-like object
#     if file.filename.endswith('.csv'):
#         data = pd.read_csv(file_like_object)
#         print(data)
#     elif file.filename.endswith(('.xls', '.xlsx')):
#         data = pd.read_excel(file_like_object)
#     else:
#         raise HTTPException(status_code=400, detail="File type not supported")

#     if target_column not in data.columns:
#         raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data.")

#     # Check for null values
#     if data.isnull().values.any():
#         raise HTTPException(status_code=400, detail="Data contains null values.")

#     # Check for string values
#     string_columns = data.select_dtypes(include=['object']).columns
#     if len(string_columns) > 0:
#         raise HTTPException(status_code=400, detail="Data contains string values.")
     
    
#     csv_filee = file.filename.split("/")
#     print(csv_filee)
#     print(csv_filee[-1].replace(".csv","_table"))
#     file_csv=csv_filee[-1].replace(".csv","_table")
#     print("save file calling")
#     print(file_csv)
#     save_file_processed(data, file_csv)
    
#     # Create polynomial features
#     X_poly = PolynomialFeatures(degree=degree).fit_transform(data.drop(columns=[target_column]))

#     y = data[target_column]

#     X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

#     model = LinearRegression()
#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     mae = np.mean(np.abs(y_test - y_pred))
#     rmse = np.sqrt(mse)
#     r2 = model.score(X_test, y_test)

#     return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}
