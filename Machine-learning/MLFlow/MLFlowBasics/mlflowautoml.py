import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import h2o
from h2o.automl import H2OAutoML
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize H2O
h2o.init()

# Convert data to H2OFrame
train = h2o.H2OFrame(list(X_train) + [str(x) for x in y_train], column_names=iris.feature_names + ["target"])
test = h2o.H2OFrame(list(X_test) + [str(x) for x in y_test], column_names=iris.feature_names + ["target"])

# Identify predictors and response
x = train.columns[:-1]  # Exclude the last column which is the response variable
y = "target"  # Response variable

# Train AutoML model
with mlflow.start_run():
    automl = H2OAutoML(max_models=10, seed=42)
    automl.train(x=x, y=y, training_frame=train)

    # Log parameters
    mlflow.log_param("max_models", 10)
    mlflow.log_param("seed", 42)

    # Log model
    mlflow.sklearn.log_model(automl.leader, "AutoML_Model")

# Stop H2O
h2o.cluster().shutdown()
