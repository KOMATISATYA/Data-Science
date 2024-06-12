import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
mlflow.start_run(run_name='random_forest_experiment')

# Train RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Log parameters
mlflow.log_param('n_estimators', 100)

# Log model
mlflow.sklearn.log_model(clf, 'random_forest_model')

# Evaluate model
accuracy = clf.score(X_test, y_test)

# Log metrics
mlflow.log_metric('accuracy', accuracy)

# End MLflow run
mlflow.end_run()