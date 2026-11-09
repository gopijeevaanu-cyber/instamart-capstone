
import sqlite3
import csv
import os

# ------------------------------------------------
# DATABASE LOCATION
# ------------------------------------------------

DB_PATH = "instamart.db"
DATA_PATH = os.path.join("data", "raw")

# Connect to SQLite database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# ------------------------------------------------
# CREATE TABLES
# ------------------------------------------------

# Remove existing tables when rebuilding the database
cursor.execute("DROP TABLE IF EXISTS orders")
cursor.execute("DROP TABLE IF EXISTS customers")
cursor.execute("DROP TABLE IF EXISTS products")

# Customers table
cursor.execute("""
CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    city TEXT,
    signup_date TEXT,
    age TEXT,
    gender TEXT
)
""")

# Products table
cursor.execute("""
CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    product_name TEXT,
    category TEXT,
    price_inr INTEGER
)
""")

# Orders table
# A separate row ID allows duplicate order IDs
# in the raw dataset to be preserved.
cursor.execute("""
CREATE TABLE orders (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT,
    customer_id TEXT,
    product_id TEXT,
    quantity TEXT,
    order_ts TEXT,
    status TEXT,
    discount_pct TEXT,
    order_amount TEXT,
    delivery_minutes TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
)
""")

# ------------------------------------------------
# IMPORT CSV DATA
# ------------------------------------------------

def import_csv(filename, table_name, columns):
    filepath = os.path.join(DATA_PATH, filename)

    with open(filepath, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        placeholders = ", ".join(["?"] * len(columns))
        column_names = ", ".join(columns)

        query = f"""
        INSERT INTO {table_name} ({column_names})
        VALUES ({placeholders})
        """

        rows = [
            tuple(row.get(column) for column in columns)
            for row in reader
        ]

        cursor.executemany(query, rows)
        print(f"{table_name}: {len(rows)} records imported")


# Import customers
import_csv(
    "customers.csv",
    "customers",
    ["customer_id", "city", "signup_date", "age", "gender"]
)

# Import products
import_csv(
    "products.csv",
    "products",
    ["product_id", "product_name", "category", "price_inr"]
)

# Import orders
import_csv(
    "orders.csv",
    "orders",
    [
        "order_id",
        "customer_id",
        "product_id",
        "quantity",
        "order_ts",
        "status",
        "discount_pct",
        "order_amount",
        "delivery_minutes"
    ]
)

# ------------------------------------------------
# SAVE DATABASE
# ------------------------------------------------

conn.commit()

print("\nDatabase created successfully!")

# Display table names
cursor.execute("""
SELECT name FROM sqlite_master
WHERE type = 'table'
""")

print("\nTables created:")
for table in cursor.fetchall():
    print(table[0])

conn.close()
