from fastapi import FastAPI, File, UploadFile
import asyncpg

app = FastAPI()

async def save_csv_to_db(file_contents):
    # Connect to PostgreSQL
    conn = await asyncpg.connect(user="postgres", password="Sbksatya@1919", database="Sales_Product", host="localhost")

    try:
        # Save CSV file contents to database
        await conn.execute("INSERT INTO csv_files (file_data) VALUES ($1)", file_contents)
    finally:
        # Close connection
        await conn.close()

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File('C:/Users/Komati Satya/MlBasics/Csv files/titanic.csv')):
    # Check if uploaded file is CSV
    if file.filename.endswith(".csv"):
        # Read file contents
        contents = await file.read()

        # Save CSV file contents to database
        await save_csv_to_db(contents)

        return {"message": "CSV file uploaded and saved in database successfully"}
    else:
        return {"error": "Uploaded file is not a CSV"}
