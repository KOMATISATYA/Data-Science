# Import H2O and train a model (example)
import h2o
from h2o.estimators import H2ORandomForestEstimator
import mlflow
h2o.init()
data = h2o.import_file("https://s3.amazonaws.com/erin-data/higgs/higgs_train_10k.csv")
x = data.columns
y = "response"
x.remove(y)
model = H2ORandomForestEstimator(ntrees=10, max_depth=5)
model.train(x=x, y=y, training_frame=data)

# Define the path where you want to save the model
save_path = "path/to/save/the/model"

# Save the H2O model using the mlflow.h2o.save_model function
mlflow.h2o.save_model(
    h2o_model=model,
    path=save_path
)
# Log the H2O model as an MLflow artifact
mlflow.h2o.log_model(
    h2o_model=model,
    artifact_path="h2o_model",
    registered_model_name="my_registered_model"
)
