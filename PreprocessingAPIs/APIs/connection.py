# import mysql.connector
# import pandas as pd
# conn = mysql.connector.connect(
#     database = "sys",
#     user = "root",
#     password = "root",
#     host = "localhost",
#     port = "3306"
# )

# conn.autocommit = True
# dataset = pd.DataFrame({'Names': ['Abhinav', 'Aryan',
#                                   'Manthan'],
#                         'DOB': ['10/01/2009', '24/03/2009',
#                                 '28/02/2009']})
# # show the dataframe
# print(dataset)
# conn = mysql.connector.connect(
#     user = "root",
#     password = "root",
#     host = "localhost",
#     port = "3306"
# )

# conn.autocommit = True

# cursor = conn.cursor()

# database_name = 'mldb'

# create_query = f"CREATE DATABASE IF NOT EXISTS {database_name}"

# cursor.execute(create_query)
# query2=f"use {database_name}"
# cursor.execute(query2)



# conn.commit()

# conn.close

'''
import mysql.connector
import pandas as pd

# Connect to MySQL database
conn = mysql.connector.connect(
    user="root",
    password="root",
    host="localhost",
    port="3306",
    database="mldb"  # Change to your desired database name
)

# Enable autocommit mode
conn.autocommit = True

# Create DataFrame
dataset = pd.DataFrame({'Names': ['Abhinav', 'Aryan', 'Manthan'],
                        'DOB': ['10/01/2009', '24/03/2009', '28/02/2009']})

# Show the DataFrame
print(dataset)

# Get column names
columns = ', '.join(dataset.columns)

# Create a SQL CREATE TABLE statement
create_table_query = f"CREATE TABLE IF NOT EXISTS employee_data ({', '.join([f'{col} VARCHAR(255)' for col in dataset.columns])})"

# Execute CREATE TABLE statement
cursor = conn.cursor()
cursor.execute(create_table_query)

# Insert data into the table
for _, row in dataset.iterrows():
    insert_query = f"INSERT INTO employee_data ({columns}) VALUES ({', '.join([f'{val!r}' for val in row])})"
    cursor.execute(insert_query)

# Commit changes
conn.commit()

# Close cursor and connection
cursor.close()
conn.close()
'''

path = "C:/Users/Komati Satya/MlBasics/Csv files/titanic.csv"
csv_file = path.split("/")
print(csv_file[-1].replace(".csv","_table"))
