# # import mlflow
# # import h2o
# # from h2o.automl import H2OAutoML
# # from sklearn.model_selection import train_test_split

# # # Initialize H2O
# # h2o.init()

# # # Load data
# # data = h2o.import_file("C:\\Users\\Komati Satya\\Data Science\\MlBasics\\Csv files\\titanic.csv")


# # # Split data into train and test sets
# # train, test = data.split_frame(ratios=[0.8], seed=1)

# # # Specify model settings
# # aml_settings = {
# #     "max_runtime_secs": 60,
# #     "seed": 1
# # }

# # # Define features (all columns except the target)
# # x = data.columns
# # y = "Survived"
# # x.remove(y)

# # # Train AutoML model
# # automl = H2OAutoML(**aml_settings)
# # automl.train(x=x, y=y, training_frame=train)


# # # End MLflow run
# # mlflow.end_run()
# import mlflow
# import h2o
# from h2o.automl import H2OAutoML
# from sklearn.model_selection import train_test_split

# # Initialize MLflow
# mlflow.start_run()

# # Initialize H2O
# h2o.init()

# # Load data
# data = h2o.import_file("C:\\Users\\Komati Satya\\Data Science\\MlBasics\\Csv files\\titanic.csv")

# # Split data into train and test sets
# train, test = data.split_frame(ratios=[0.8], seed=1)

# # Specify model settings
# aml_settings = {
#     "max_runtime_secs": 60,
#     "seed": 1
# }

# # Define features (all columns except the target)
# x = data.columns
# y = "Survived"
# x.remove(y)

# # Train AutoML model
# automl = H2OAutoML(**aml_settings)
# automl.train(x=x, y=y, training_frame=train)

# # Log AutoML model
# mlflow.h2o.log_model(automl.leader, "h2o_automl_model")

# # Log metrics
# if automl.leader.scoring_history() is not None:
#     if "logloss" in automl.leader.scoring_history():
#         mlflow.log_metric("logloss", automl.leader.logloss())

# # Log parameters
# mlflow.log_params(aml_settings)

# # End MLflow run
# mlflow.end_run()



# 


# import mlflow
# import h2o
# from h2o.automl import H2OAutoML
# from sklearn.model_selection import train_test_split

# # Initialize H2O
# h2o.init()

# # Load data
# data = h2o.import_file("C:\\Users\\Komati Satya\\Data Science\\MlBasics\\Csv files\\titanic.csv")

# # Split data into train and test sets
# train, test = data.split_frame(ratios=[0.8], seed=1)

# # Specify model settings
# aml_settings = {
#     "max_runtime_secs": 60,
#     "seed": 1
# }

# # Define features (all columns except the target)
# x = data.columns
# y = "Survived"
# x.remove(y)

# # Train AutoML model
# automl = H2OAutoML(**aml_settings)
# automl.train(x=x, y=y, training_frame=train)
# # best_model = automl.leader
# # Log AutoML model
# with mlflow.start_run():
#     mlflow.log_params(automl.leader.params)  # Log model parameters
#     mlflow.log_metric("RMSE", automl.leader.rmse())  # Log RMSE score
#     mlflow.log_metric("MSE", automl.leader.mse())  # Log MSE score
#     mlflow.log_metric("R2", automl.leader.r2())  # Log R2 score
#     mlflow.h2o.log_model(automl.leader, "h2o_automl_model")  # Log H2O model
# mlflow.h2o.log_model(best_model, "h2o_automl_model")
# # End MLflow run
# mlflow.end_run()

import mlflow
import h2o
from h2o.automl import H2OAutoML
from sklearn.model_selection import train_test_split

# Initialize H2O
h2o.init()

# Load data
data = h2o.import_file("C:\\Users\\Komati Satya\\Data Science\\MlBasics\\Csv files\\titanic.csv")

# Split data into train and test sets
train, test = data.split_frame(ratios=[0.8], seed=1)

# Specify model settings
aml_settings = {
    "max_runtime_secs": 60,
    "seed": 1
}

# Define features (all columns except the target)
x = data.columns
y = "Survived"
x.remove(y)

# Train AutoML model
automl = H2OAutoML(**aml_settings)
automl.train(x=x, y=y, training_frame=train)

# Log AutoML model
with mlflow.start_run():
    mlflow.log_params(automl.leader.params)  # Log model parameters
    mlflow.log_metric("RMSE", automl.leader.rmse())  # Log RMSE score
    mlflow.log_metric("MSE", automl.leader.mse())  # Log MSE score
    mlflow.log_metric("R2", automl.leader.r2())  # Log R2 score
mlflow.h2o.log_model(automl.leader, "h2o_automl_model")  # Log H2O model
print(automl.leader.params)
print(automl.leader)
# End MLflow run
mlflow.end_run()

