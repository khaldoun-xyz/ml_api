from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# --- 1. Load and Prepare Data ---
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


# --- 2. Preprocessing & Feature Engineering ---

# A) Define the Target (y)
target_map = {"True": 1, True: 1, "False": 0, False: 0}
df["loan_approved_numeric"] = df["loan_approved"].map(target_map)
y = df["loan_approved_numeric"]

# B) Define the Features (X)
# We drop the original target columns and the non-predictive 'name' column.
# We also drop 'city' because it has too many unique values (high cardinality)
# and would require complex feature engineering (like target encoding).
# Our EDA showed it wasn't a strong predictor anyway.
X = df.drop(columns=["loan_approved", "loan_approved_numeric", "name", "city"])

print("Features being used for training:")
print(X.head())


# --- 3. Split Data into Training and Testing Sets ---

# We split the data 80% for training and 20% for testing.
# - random_state=42 ensures you get the same split every time you run this.
# - stratify=y ensures that both your train and test sets have the
#   same 50/50 balance of approvals/rejections as the original dataset.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")


# --- 4. Train the Classification Model ---

print("\n--- Training Model ---")
print("Training a RandomForestClassifier...")

# Initialize the model
# n_estimators=100 means it will build 100 "decision trees"
model = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)

# Train the model on the training data
model.fit(X_train, y_train)

print(f"Model trained successfully.")
print(
    f"Model OOB Score: {model.oob_score_:.4f}"
)  # Out-of-bag score, a quick accuracy check
# A Random Forest builds many individual decision trees.
# Each tree is trained on a random sample of your training data (called a "bootstrap sample").
# Because of this random sampling, each tree never sees about 1/3 of the training data.
# This "left out" data is called its Out-of-Bag (OOB) sample.
# To get the OOB score, the model takes each data point and has only the trees
# that never saw it vote on a prediction.
# It mimics how the model will perform on new, unseen data.

# --- 5. Evaluate the Model ---

print("\n--- Model Evaluation ---")

# Make predictions on the unseen test data
y_pred = model.predict(X_test)

# A) Accuracy
# Because our dataset is perfectly 50/50 balanced, accuracy is a good metric.
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on Test Set: {accuracy * 100:.2f}%")

# B) Classification Report
# This shows precision, recall, and f1-score for both classes (0=Rejected, 1=Approved)
print("\nClassification Report:")
# target_names=['Rejected', 'Approved'] will make the report easier to read
print(
    classification_report(y_test, y_pred, target_names=["Rejected (0)", "Approved (1)"])
)

# C) Confusion Matrix
# This shows us exactly what we got right and wrong.
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix for a clearer view
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
plt.show()


# --- 6. (Bonus) Check Feature Importance ---

print("\n--- Feature Importance ---")
# See which features the model found most predictive

importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print(importances)

# Plot feature importances
plt.figure(figsize=(10, 5))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()
