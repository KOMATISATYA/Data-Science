import mysql.connector


def sql_connection():
    conn = mysql.connector.connect(
    user = "root",
    password = "root",
    host = "localhost",
    port = "3306",
    autocommit = True
)
    cursor = conn.cursor()
    return cursor


def create_db(db_name):
    cursor=sql_connection()
    database_name = db_name
    create_query = f"CREATE DATABASE IF NOT EXISTS {database_name}"
    cursor.execute(create_query)
    query2=f"use {database_name}"
    cursor.execute(query2)
   
    result = cursor.fetchall()
    print(result)
   
    return result

def sql_execute(query):
    cursor=sql_connection()
    cursor.execute(query)
    # conn.commit()
    return "success"
