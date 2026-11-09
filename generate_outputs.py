
import os
import re
import pandas as pd
import numpy as np
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------
# 1. FOLDER PATHS
# -----------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data", "clean")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------
# 2. LOAD CLEAN DATA
# -----------------------------------

customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
orders = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"))

def find_column(df, candidates):
    columns = {
        re.sub(r"[^a-z0-9]", "", str(col).lower()): col
        for col in df.columns
    }

    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in columns:
            return columns[key]

    return None

customer_id = find_column(
    customers, ["customer_id", "customerid", "user_id"]
)

order_customer_id = find_column(
    orders, ["customer_id", "customerid", "user_id"]
)

order_date = find_column(
    orders, ["order_date", "created_at", "date", "order_timestamp", "order_ts"]
)

amount_column = find_column(
    orders, ["total_amount", "order_amount", "amount",
             "order_value", "total", "grand_total", "order_amt"]
)

status_column = find_column(
    orders, ["status", "order_status"]
)

if customer_id is None or order_customer_id is None:
    raise ValueError(
        "Customer ID column not found. "
        "Check the column names in customers.csv and orders.csv."
    )

if order_date is None:
    raise ValueError(
        "Order date column not found in orders.csv."
    )

# Standardize IDs for merging
customers[customer_id] = customers[customer_id].astype(str)
orders[order_customer_id] = orders[order_customer_id].astype(str)

orders[order_date] = pd.to_datetime(
    orders[order_date], errors="coerce"
)

orders = orders.dropna(subset=[order_date])

# Exclude cancelled and refunded orders if status exists
if status_column:
    excluded = ["cancelled", "canceled", "refunded"]
    orders = orders[
        ~orders[status_column].astype(str).str.lower().isin(excluded)
    ]

# Get purchase amount
if amount_column:
    orders[amount_column] = pd.to_numeric(
        orders[amount_column], errors="coerce"
    ).fillna(0)
    orders["_amount"] = orders[amount_column]
else:
    print("WARNING: No order amount column found. Using zero monetary value.")
    orders["_amount"] = 0

if orders.empty:
    raise ValueError("No valid dated orders found in orders.csv.")

# -----------------------------------
# 3. CUSTOMER PURCHASE FEATURES
# -----------------------------------

reference_date = orders[order_date].max()

features = orders.groupby(order_customer_id).agg(
    frequency=(order_date, "count"),
    monetary=("_amount", "sum"),
    last_order=(order_date, "max"),
    first_order=(order_date, "min")
).reset_index()

features = features.rename(
    columns={order_customer_id: customer_id}
)

features["recency"] = (
    reference_date - features["last_order"]
).dt.days

features["tenure_days"] = (
    features["last_order"] - features["first_order"]
).dt.days

features["average_order_value"] = (
    features["monetary"] / features["frequency"].clip(lower=1)
)

# Include customers who have never ordered
customer_data = customers.drop_duplicates(
    subset=[customer_id]
).copy()

customer_data = customer_data.merge(
    features, on=customer_id, how="left"
)

numeric_columns = [
    "frequency", "monetary", "recency",
    "tenure_days", "average_order_value"
]

customer_data[numeric_columns] = (
    customer_data[numeric_columns].fillna(0)
)

# -----------------------------------
# 4. CUSTOMER SEGMENTATION
# -----------------------------------

segment_features = customer_data[
    ["frequency", "monetary", "recency"]
].copy()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(segment_features)

number_of_clusters = min(4, len(customer_data))

if number_of_clusters >= 2:
    kmeans = KMeans(
        n_clusters=number_of_clusters,
        random_state=42,
        n_init=10
    )

    customer_data["cluster"] = kmeans.fit_predict(
        scaled_features
    )

    cluster_stats = customer_data.groupby("cluster").agg(
        frequency=("frequency", "mean"),
        recency=("recency", "mean")
    )

    ordered_clusters = cluster_stats.sort_values(
        ["frequency", "recency"],
        ascending=[False, True]
    ).index.tolist()

    segment_names = [
        "Loyal Customers",
        "Regular Customers",
        "At Risk Customers",
        "Low Activity Customers"
    ]

    segment_mapping = {
        cluster: segment_names[i]
        for i, cluster in enumerate(ordered_clusters)
    }

    customer_data["segment"] = (
        customer_data["cluster"].map(segment_mapping)
    )

else:
    customer_data["cluster"] = 0
    customer_data["segment"] = "General Customers"

customer_data.to_csv(
    os.path.join(OUTPUT_DIR, "customer_segments.csv"),
    index=False
)

print("Customer segments generated!")

# -----------------------------------
# 5. LAPSE PREDICTION MODEL
# -----------------------------------

# Define lapse as no purchase in the last 30 days
customer_data["lapse"] = (
    customer_data["recency"] > 30
).astype(int)

model_features = [
    "frequency",
    "monetary",
    "tenure_days",
    "average_order_value"
]

X = customer_data[model_features]
y = customer_data["lapse"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X, y)

model_bundle = {
    "model": model,
    "features": model_features,
    "lapse_definition": "No purchase in the last 30 days",
    "reference_date": str(reference_date)
}

joblib.dump(
    model_bundle,
    os.path.join(OUTPUT_DIR, "lapse_model.joblib")
)

print("Lapse prediction model generated!")

# -----------------------------------
# 6. RETENTION LIST
# -----------------------------------

probabilities = model.predict_proba(X)

if 1 in model.classes_:
    risk_index = list(model.classes_).index(1)
    customer_data["lapse_risk"] = probabilities[:, risk_index]
else:
    customer_data["lapse_risk"] = 0.0

retention_list = customer_data[
    customer_data["lapse_risk"] >= 0.5
].copy()

retention_list = retention_list.sort_values(
    "lapse_risk", ascending=False
)

retention_list.to_csv(
    os.path.join(OUTPUT_DIR, "retention_list.csv"),
    index=False
)

print("Retention list generated!")

# -----------------------------------
# 7. FINAL SUMMARY
# -----------------------------------

print("\n================================")
print("ALL OUTPUT FILES GENERATED!")
print("================================")

print("Customer segments:",
      len(customer_data))

print("Retention customers:",
      len(retention_list))

print("\nFiles saved inside:", OUTPUT_DIR)

print("\n1. customer_segments.csv")
print("2. lapse_model.joblib")
print("3. retention_list.csv")
