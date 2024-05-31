# import pandas as pd
# import mysql.connector

# conn=mysql.connector.connect(
#     user='root',
#     password='root',
#     port='3306',
#     database='mlclassification',
#     host = "localhost",
# )

# # cursor=conn.cursor()

# # cursor.execute("select * from heart_failure_table ")

# # cursor.fetchall()

# def mysql_to_dataframe(table_name):
#     """
#     Retrieve data from a MySQL table and convert it into a DataFrame.
    
#     Args:
#     - table_name: The name of the MySQL table from which data is to be retrieved.
    
#     Returns:
#     - df: DataFrame containing the data from the MySQL table.
#     """
#     try:
         
#         # Connect to the MySQL database
#         # with engine.connect() as conn:
#             # Execute SQL query to select all data from the specified table
#          query = f"SELECT * FROM {table_name}"
#             # Use pandas' read_sql_query to retrieve data into a DataFrame
#          df = pd.read_sql_query(query, conn)
#          return df
#     except Exception as e:
#         print("Error:", e)
#         return None

# # Example usage:
# if __name__ == "__main__":
#     table_name = "heart_failure_table"  # Replace 'your_table_name' with the name of your table
#     df = mysql_to_dataframe(table_name)
#     if df is not None:
#         print("DataFrame:")
#         print(df.head())
#     else:
#         print("Failed to retrieve data from MySQL table.")


# from sklearn.preprocessing import OrdinalEncoder

# def ordinal_encoding(df):
#     # Create an empty dictionary to store the order of categories for each categorical column
#     order_dict = {}

#     # Iterate over each column in the DataFrame
#     for col in df.columns:
#         print("Processing column:", col)
#         # Check if the column is categorical (dtype == 'object')
#         if df[col].dtype == 'object':
#             print("Column", col, "is categorical")
#             # Extract unique categories and sort them
#             categories = df[col].unique()
#             categories.sort()
#             # Store the sorted categories in the order_dict
#             order_dict[col] = list(categories)
#             print("Categories for column", col, ":", order_dict[col])
#         else:
#             print("Column", col, "is not categorical, adding to order_dict with unique values...")
#             order_dict[col] = df[col].unique().tolist()

#     # Check if any categorical columns exist
#     if not order_dict:
#         raise ValueError("No categorical columns found for ordinal encoding")

#     # Create an instance of OrdinalEncoder with the order_dict
#     ordinal_encoder = OrdinalEncoder(categories=[order_dict[col] for col in df.columns if col in order_dict])
    
#     # Extract categorical columns
#     categorical_cols = [col for col in df.columns if col in order_dict]
#     # Extract numeric columns
#     numeric_cols = [col for col in df.columns if col not in categorical_cols]

#     # Perform ordinal encoding on the categorical columns
#     encoded_categorical_data = ordinal_encoder.fit_transform(df[categorical_cols])

#     # Combine numeric columns with encoded categorical columns
#     encoded_data = pd.concat([df[numeric_cols], pd.DataFrame(encoded_categorical_data, columns=categorical_cols)], axis=1)

#     # Convert the encoded data to a list of dictionaries
#     encoded_data_dict = encoded_data.to_dict(orient='records')

#     return {"encoded_data": encoded_data_dict}

# # Example usage:
# import pandas as pd

# # Create a sample DataFrame
# data = {
#     'Name': ['Andrea', 'Angelina', 'Arnold', 'Brad', 'David', 'Donald', 'Gautam', 'Ismail', 'Kory', 'Michael', 'Mohan', 'Rob', 'Tom'],
#     'Age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85],
#     'Income($)': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000, 150000, 160000, 170000],
#     'Class': ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D', 'A', 'B', 'C', 'D', 'A']
# }
# df = pd.DataFrame(data)

# # Perform ordinal encoding
# ordinal_encoded_result = ordinal_encoding(df)
# print("Ordinal Encoding Result:")
# print(ordinal_encoded_result)


from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from typing import List

app = FastAPI()

def load_data(filename):
    data = pd.read_csv(filename)
    return data

def hierarchical_clustering(data, n_clusters=None, distance_threshold=None):
    model = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=distance_threshold)
    clusters = model.fit_predict(data)
    clustered_data = []
    for i, data_point in enumerate(data.to_dict(orient='records')):
            data_point['cluster_label'] = int(clusters[i])  # Ensure cluster label is converted to int
            clustered_data.append(data_point)
    return clustered_data

@app.post("/cluster/")
async def cluster_data(file: UploadFile = File(...)):
    try:
        # Load data from CSV
        data = load_data(file.file)
        
        # Perform clustering
        n_clusters = None  # Set the number of clusters, or None for distance_threshold based clustering
        distance_threshold = 10  # Set the distance threshold for clustering
        clusters = hierarchical_clustering(data, n_clusters=n_clusters, distance_threshold=distance_threshold)
        
        # Return cluster assignments
        return clusters
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# from collections import Counter
# s=[4, 7, 8, 3, 7, 5, 3, 4]
# dup = []
# for i in s:
#     if i not in dup:
#         dup.append(i)
# print(dup)
# print(s)
# t=set(s)
# print(t)
# q=list(t)
# print(q)
# count = Counter(s) 
# print(list(count))