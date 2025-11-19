import sqlite3

conn = sqlite3.connect("churn.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER,
    state TEXT,
    device TEXT,
    gender TEXT,
    tenure INTEGER,
    subscription_plan TEXT,
    unit_price REAL,
    times_purchased INTEGER,
    revenue REAL,
    data_usage REAL,
    churn_prediction INTEGER,
    probability REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()

print("Database + Table created successfully!")
