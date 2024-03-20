import mysql.connector

def sql_connection():
    conn = mysql.connector.connect(
        user="root",
        password="root",
        host="localhost",
        port="3306",
        autocommit=True
    )
    cursor = conn.cursor()
    return conn, cursor

def close_connection(conn, cursor):
    cursor.close()
    conn.close()

def create_db(db_name):
    conn, cursor = sql_connection()
    create_query = f"CREATE DATABASE IF NOT EXISTS {db_name}"
    cursor.execute(create_query)
    cursor.execute(f"USE {db_name}")
    close_connection(conn, cursor)

def sql_execute(query, db_name):
    conn, cursor = sql_connection()
    cursor.execute(f"USE {db_name}") 
    cursor.execute(query)
    close_connection(conn, cursor)
    return "success"


create_db("mllll")

queryy = """CREATE TABLE Persons (
    PersonID int,
    LastName varchar(255),
    FirstName varchar(255),
    Address varchar(255),
    City varchar(255)
);"""

sql_execute(queryy, "mllll") 