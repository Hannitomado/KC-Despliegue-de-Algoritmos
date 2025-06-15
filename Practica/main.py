import argparse
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from utils import prepare_data

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Text classification with MLflow")
parser.add_argument("--model", type=str, choices=["logreg", "nb"], default="logreg", help="Model to use")
parser.add_argument("--max_features", type=int, default=5000, help="Number of TF-IDF features")

args = parser.parse_args()

# Load dataset
data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

# Preprocess
X_train, X_test, y_train, y_test, vectorizer = prepare_data(data.data, data.target, max_features=args.max_features)

# Select model
if args.model == "logreg":
    model = LogisticRegression(max_iter=1000)
elif args.model == "nb":
    model = MultinomialNB()

# Start MLflow run
with mlflow.start_run():
    mlflow.log_param("model_type", args.model)
    mlflow.log_param("max_features", args.max_features)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)

    # Log model
    mlflow.sklearn.log_model(model, "model")
