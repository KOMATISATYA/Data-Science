# from fastapi import FastAPI, File, UploadFile
# import pandas as pd
# from io import BytesIO
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from category_encoders import BinaryEncoder, WOEEncoder, HashingEncoder, BaseNEncoder, TargetEncoder

# app = FastAPI()



# def read_file(file: UploadFile):
#     contents = file.file.read()
#     return pd.read_csv(BytesIO(contents))

# @app.post("/label_encoding")
# async def label_encoding(file: UploadFile):
#     df = read_file(file)
#     encoder = LabelEncoder()
#     encoded_data = df.apply(encoder.fit_transform)
#     return {"encoded_data": encoded_data.to_dict(orient='records')}

# @app.post("/one_hot_encoding")
# async def one_hot_encoding(file: UploadFile):
#     df = read_file(file)
#     encoder = OneHotEncoder(sparse=False)
#     encoded_data = encoder.fit_transform(df)
#     return {"encoded_data": encoded_data.tolist()}

# @app.post("/dummy_encoding")
# async def dummy_encoding(file: UploadFile):
#     df = read_file(file)
#     encoder = BinaryEncoder()
#     encoded_data = encoder.fit_transform(df)
#     return {"encoded_data": encoded_data.to_dict(orient='records')}

# @app.post("/effect_encoding")
# async def effect_encoding(file: UploadFile, target_file: UploadFile):
#     df = read_file(file)
#     target_df = read_file(target_file)
#     encoder = WOEEncoder()
#     encoded_data = encoder.fit_transform(df, target_df)
#     return {"encoded_data": encoded_data.to_dict(orient='records')}

# @app.post("/hash_encoding")
# async def hash_encoding(file: UploadFile):
#     df = read_file(file)
#     encoder = HashingEncoder()
#     encoded_data = encoder.fit_transform(df)
#     return {"encoded_data": encoded_data.to_dict(orient='records')}

# @app.post("/binary_encoding")
# async def binary_encoding(file: UploadFile):
#     df = read_file(file)
#     encoder = BinaryEncoder()
#     encoded_data = encoder.fit_transform(df)
#     return {"encoded_data": encoded_data.to_dict(orient='records')}

# @app.post("/base_n_encoding")
# async def base_n_encoding(file: UploadFile):
#     df = read_file(file)
#     encoder = BaseNEncoder()
#     encoded_data = encoder.fit_transform(df)
#     return {"encoded_data": encoded_data.to_dict(orient='records')}

# @app.post("/target_encoding")
# async def target_encoding(file: UploadFile, target_file: UploadFile):
#     df = read_file(file)
#     target_df = read_file(target_file)
#     encoder = TargetEncoder()
#     encoded_data = encoder.fit_transform(df, target_df)
#     return {"encoded_data": encoded_data.to_dict(orient='records')}
