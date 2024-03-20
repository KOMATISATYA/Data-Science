# import pandas as pd
# from scipy.stats.mstats import winsorize
# import numpy as np
# from scipy.stats import scoreatpercentile

# target_url = 'C:/Users/Komati Satya/MlBasics/Csv files/House.csv'

# df = pd.read_csv(target_url)
# print(df.dtypes)



# # def remove_outliers_winsorization(df, lower_limit=0.15, upper_limit=0.20):
# #     for column in df_numeric:
# #         print(column)
# #         winsorized_values = winsorize(df[column], limits=(lower_limit, upper_limit))
# #         df[column] = winsorized_values
# #     return df
# # cleaned_df = remove_outliers_winsorization(df)
# # print(cleaned_df)

# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import FileResponse
# import pandas as pd
# import tempfile

# app = FastAPI()

# # Preprocessing function
# def preprocess_data(input_df):
#     # Your preprocessing steps here
#     # Example: Filling missing values
#     input_df.fillna(0, inplace=True)
#     # Example: Scaling
#     input_df = (input_df - input_df.min()) / (input_df.max() - input_df.min())
#     # Example: Outlier removal
#     # Implement outlier removal logic as per your requirements
#     return input_df

# @app.post("/preprocess")
# async def preprocess(file: UploadFile = File(...)):
#     # Check if file is CSV
#     if not file.filename.endswith('.csv'):
#         raise HTTPException(status_code=400, detail="Only CSV files are supported")

#     # Read the raw file data into a DataFrame
#     with tempfile.NamedTemporaryFile(delete=False) as temp:
#         temp.write(await file.read())
#         temp_name = temp.name
    
#     input_df = pd.read_csv(temp_name)

#     # Preprocess the data
#     preprocessed_df = preprocess_data(input_df)
    
#     # Convert preprocessed DataFrame to CSV
#     csv_data = preprocessed_df.to_csv(index=False)

#     # Return CSV file as response
#     return FileResponse(temp_name, filename='preprocessed_data.csv')



from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse
import pandas as pd
import tempfile
import io

app = FastAPI()

# Preprocessing function
def preprocess_data(input_df):
    # Your preprocessing steps here
    # Example: Filling missing values for numerical columns
    numerical_columns_to_fill = input_df.columns[input_df.dtypes != 'object']
    input_df[numerical_columns_to_fill] = input_df[numerical_columns_to_fill].fillna(0)
    # Example: Scaling numerical columns
    for column in numerical_columns_to_fill:
        input_df[column] = (input_df[column] - input_df[column].min()) / (input_df[column].max() - input_df[column].min())
    # Example: Handle outliers for numerical columns
    # Implement outlier removal logic as per your requirements for numerical columns
    return input_df.to_csv(index=False)

@app.post("/preprocess")
async def preprocess(file: UploadFile = File(...), string_input: str = Form(...)):
    # Check if file is CSV
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    # Read the raw file data into a DataFrame
    contents = await file.read()
    input_df = pd.read_csv(io.BytesIO(contents))
    
    # Preprocess the data
    preprocessed_csv = preprocess_data(input_df)
    print(string_input)
    # Return CSV file as response
    return StreamingResponse(io.BytesIO(preprocessed_csv.encode()), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=preprocessed_data.csv"})

# import pandas library
# import pandas as pd
 
# # create a dataframe
# # object from dictionary
# dataset = pd.DataFrame({'Names': ['Abhinav', 'Aryan',
#                                   'Manthan'],
#                         'DOB': ['10/01/2009', '24/03/2009',
#                                 '28/02/2009']})
# # show the dataframe
# print(dataset)

# # importing sql library
# from sqlalchemy import create_engine
 
# # create a reference
# # for sql library
# engine = create_engine('sqlite://',
#                        echo=False)
 
# # attach the data frame to the sql
# # with a name of the table
# # as "Employee_Data"
# dataset.to_sql('Employee_Data',
#                con=engine)
 
# # create a connection
# conn = engine.connect()

# # show the complete data
# # from Employee_Data table
# print(conn.execute("SELECT * FROM Employee_Data").fetchall())

# # close the connection
# conn.close()


# import mysql.connector
# import csv

# # MySQL connection parameters
# host = 'localhost'
# user = 'root'
# password = 'root'
# database = 'mydb'

# # CSV file path
# csv_file = 'C:/Users/Komati Satya/MlBasics/Csv files/'
# file_name = "titanic.csv"

# file_path = csv_file + file_name

# # Modify the table name variable to avoid conflicts with reserved keywords
# table_name = file_name.replace('.csv', '_table')
# table_name = table_name.replace("-", "_")
# print(table_name)

# # Connect to MySQL
# conn = mysql.connector.connect(host=host, user=user, password=password, database=database)
# cursor = conn.cursor()

# # Open CSV file and read header
# with open(file_path, 'r') as file:
#     reader = csv.reader(file)
#     header = next(reader)  # Read the header row

# # Create table if not exists
# create_table_query = f"CREATE TABLE IF NOT EXISTS `{table_name}` ("
# for column in header:
#     create_table_query += f"`{column}` VARCHAR(255), "
# create_table_query = create_table_query.rstrip(', ') + ")"
# cursor.execute(create_table_query)

# # Insert data into table
# with open(file_path, 'r') as file:
#     reader = csv.DictReader(file)
#     for row in reader:
#         # Replace empty strings with None (for NULL values in MySQL)
#         row = {key: (value if value != '' else None) for key, value in row.items()}

#         # Prepare column names and values for insertion
#         columns = ', '.join(row.keys())
#         placeholders = ', '.join(['%s'] * len(row))
#         values = tuple(row.values())

#         # Insert row into table
#         insert_query = f"INSERT INTO `{table_name}` ({columns}) VALUES ({placeholders})"
#         cursor.execute(insert_query, values)

# # Commit changes to the database
# conn.commit()

# # Close cursor and connection
# cursor.close()
# conn.close()