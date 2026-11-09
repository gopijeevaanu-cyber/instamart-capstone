
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Folders
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load customer data
df = pd.read_csv(OUTPUT_DIR / "customer_segments.csv")

print("\n========== PART B MODEL EVALUATION ==========")

# Model features
features = [
    "frequency",
    "monetary",
    "tenure_days",
    "average_order_value"
]

# Check required columns
missing = [c for c in features + ["recency"] if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns in customer_segments.csv: {missing}")

# Prepare data
for col in features + ["recency"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[features + ["recency"]] = df[
    features + ["recency"]
].replace([np.inf, -np.inf], np.nan)

df = df.dropna(subset=features + ["recency"]).copy()

# Define lapse: no purchase in the last 30 days
df["lapse"] = (df["recency"] > 30).astype(int)

X = df[features]
y = df["lapse"]

print("\nTotal customers:", len(df))
print("Active customers:", int((y == 0).sum()))
print("Lapsed customers:", int((y == 1).sum()))

if y.nunique() != 2 or y.value_counts().min() < 2:
    raise ValueError(
        "Not enough examples in both classes for a valid train/test split."
    )

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train evaluation model on training data only
evaluation_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

evaluation_model.fit(X_train, y_train)

# Predict unseen test records
y_pred = evaluation_model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n========== MODEL PERFORMANCE ==========")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save evaluation metrics
metrics = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Score": [accuracy, precision, recall, f1]
})

metrics.to_csv(
    OUTPUT_DIR / "model_evaluation.csv",
    index=False
)

# Save actual and predicted test labels
pd.DataFrame({
    "actual_lapse": y_test.to_numpy(),
    "predicted_lapse": y_pred
}).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

# Train final model on all available records
final_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

final_model.fit(X, y)

# Calculate risk probabilities for all customers
df["lapse_probability"] = final_model.predict_proba(X)[:, 1]

# Create final retention list
retention = df[df["lapse_probability"] >= 0.5].copy()

retention = retention.sort_values(
    "lapse_probability", ascending=False
)

retention.to_csv(
    OUTPUT_DIR / "retention_list.csv",
    index=False
)

# Create top 10 at-risk customers
top10 = retention.head(10).copy()

top10.to_csv(
    OUTPUT_DIR / "top10_at_risk_customers.csv",
    index=False
)

# Update saved model bundle
model_bundle = {
    "model": final_model,
    "features": features,
    "lapse_definition": "No purchase in the last 30 days",
    "reference_date": "2025-09-30 22:46:26"
}

joblib.dump(
    model_bundle,
    OUTPUT_DIR / "lapse_model.joblib"
)

# Save project evaluation summary
with open(OUTPUT_DIR / "evaluation_summary.txt", "w") as f:
    f.write("INSTAMART CUSTOMER LAPSE MODEL EVALUATION\n")
    f.write("=========================================\n")
    f.write(f"Total customers: {len(df)}\n")
    f.write(f"Training records: {len(X_train)}\n")
    f.write(f"Testing records: {len(X_test)}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write("\nLapse definition: No purchase in the last 30 days.\n")
    f.write(
        "Note: Evaluation uses a held-out split of historical "
        "customer records and a proxy inactivity label. "
        "It does not establish future churn prediction performance.\n"
    )

print("\n========== OUTPUTS GENERATED ==========")
print("1. model_evaluation.csv")
print("2. test_predictions.csv")
print("3. evaluation_summary.txt")
print("4. Updated lapse_model.joblib")
print("5. Updated retention_list.csv")
print("6. top10_at_risk_customers.csv")

print("\nTop 10 at-risk customers:")
print(top10.to_string(index=False))

print("\nPART B EVALUATION COMPLETED!")
