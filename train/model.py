import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def parse_arguments():
    """Parse command-line arguments for the training script."""
    parser = argparse.ArgumentParser(
        description="Train a loan approval classification model"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="random-forest-baseline",
        help="Name of the MLflow run (default: random-forest-baseline)",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated list of features to use for training. "
        "Available features: income, credit_score, loan_amount, years_employed, points"
        "If not provided, uses all available features.",
    )
    return parser.parse_args()


# --- 0. MLflow Setup ---
mlflow.set_experiment("loan-approval-model")

# --- 1. Parse Arguments ---
args = parse_arguments()
run_name = args.run_name
custom_features = args.features

# --- 2. Load and Prepare Data ---
try:
    csv_path = (
        Path(__file__).parent.parent
        / "data"
        / "loan-approval-dataset"
        / "loan_approval.csv"
    )
    df = pd.read_csv(csv_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'loan_approval.csv' not found. Please check the file path.")
    raise FileNotFoundError("File not found")

# Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()


# --- 3. Preprocessing & Feature Engineering ---

# A) Define the Target (y)
target_map = {"True": 1, True: 1, "False": 0, False: 0}
df["loan_approved_numeric"] = df["loan_approved"].map(target_map)
y = df["loan_approved_numeric"]

# B) Define the Features (X)
# We drop the original target columns and the non-predictive 'name' column.
# We also drop 'city' because it has too many unique values (high cardinality)
# and would require complex feature engineering (like target encoding).
# Our EDA showed it wasn't a strong predictor anyway.
if custom_features:
    feature_list = [f.strip() for f in custom_features.split(",")]
    X = df[feature_list]
    print(f"Using custom features: {feature_list}")
else:
    X = df.drop(columns=["loan_approved", "loan_approved_numeric", "name", "city"])
    print(
        "Using default features (all except: loan_approved, loan_approved_numeric, name, city)"
    )

print("Features being used for training:")
print(X.head())


# --- 4. Split Data into Training and Testing Sets ---

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")


# --- 5. Train the Classification Model ---

print("\n--- Training Model ---")
print(f"Training a RandomForestClassifier with run name: {run_name}...")

# Start MLflow run
with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_param("dataset_size", len(df))
    mlflow.log_param("training_set_size", len(X_train))
    mlflow.log_param("test_set_size", len(X_test))
    mlflow.log_param("num_features", X.shape[1])
    mlflow.log_param("feature_names", ", ".join(X.columns.tolist()))
    mlflow.log_param("run_name", run_name)

    n_estimators = 100
    random_state = 42

    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, oob_score=True
    )

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("oob_score", True)

    model.fit(X_train, y_train)

    print(f"Model trained successfully.")
    print(
        f"Model OOB Score: {model.oob_score_:.4f}"
    )  # Out-of-bag score, a quick accuracy check

    mlflow.log_metric("oob_score", model.oob_score_)

    # --- 6. Evaluate the Model ---

    print("\n--- Model Evaluation ---")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on Test Set: {accuracy * 100:.2f}%")

    mlflow.log_metric("accuracy", accuracy)

    print("\nClassification Report:")
    report = classification_report(
        y_test, y_pred, target_names=["Rejected (0)", "Approved (1)"], output_dict=True
    )
    print(
        classification_report(
            y_test, y_pred, target_names=["Rejected (0)", "Approved (1)"]
        )
    )

    mlflow.log_metric("precision_rejected", report["Rejected (0)"]["precision"])
    mlflow.log_metric("recall_rejected", report["Rejected (0)"]["recall"])
    mlflow.log_metric("f1_rejected", report["Rejected (0)"]["f1-score"])
    mlflow.log_metric("precision_approved", report["Approved (1)"]["precision"])
    mlflow.log_metric("recall_approved", report["Approved (1)"]["recall"])
    mlflow.log_metric("f1_approved", report["Approved (1)"]["f1-score"])

    # C) Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Reject", "Predicted Approve"],
        yticklabels=["Actual Reject", "Actual Approve"],
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "confusion_matrix.png")

    # --- 7. (Bonus) Check Feature Importance ---

    print("\n--- Feature Importance ---")
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)
    print(importances)

    mlflow.log_dict(importances.to_dict(), "feature_importances.json")

    plt.figure(figsize=(10, 5))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "feature_importance.png")

    # --- 8. Log Model to MLflow (with Signature) ---
    print("\n--- Logging Model to MLflow ---")

    signature = infer_signature(X_test, y_pred)
    input_example = X_test.head()

    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        signature=signature,
        input_example=input_example,
    )

    # --- 8b. Register the Model ---
    print("\n--- Registering Model in MLflow Registry ---")
    model_uri = f"runs:/{run.info.run_id}/model"

    registered_model_name = "loan-approval-model"

    model_version_info = mlflow.register_model(
        model_uri=model_uri, name=registered_model_name
    )
    print(
        f"Model registered: {model_version_info.name}, Version: {model_version_info.version}"
    )

    client = mlflow.MlflowClient()
    client.update_model_version(
        name=registered_model_name,
        version=model_version_info.version,
        description=f"Model trained with run name: {run_name}",
    )

    print("\n--- MLflow Run Complete ---")
    print(f"Run ID: {run.info.run_id}")
    print(f"Experiment: {run.info.experiment_id}")
    # --- End of MLflow Run ---


# --- 9. Save the trained model (for local use) ---
# We do this *after* the run is complete.
# This .joblib file is what our `predict.py` script will use.
print("\n--- Saving Model Locally ---")
model_path = Path(__file__).parent.parent / "models" / "loan_model.joblib"
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
