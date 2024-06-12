import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
with mlflow.start_run():
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Log parameters and metrics
    mlflow.log_params({"n_estimators": clf.n_estimators, "max_depth": clf.max_depth})
    mlflow.log_metric("accuracy", clf.score(X_test, y_test))
