import sqlite3
from langchain_community.utilities import SQLDatabase

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('employees.db')

# Create a cursor object
cur = conn.cursor()

# Create a new table with a UNIQUE constraint on (name, age, department)
cur.execute('''
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER,
    department TEXT,
    UNIQUE(name, age, department)
)
''')

# Commit the changes
conn.commit()

# Insert data into the table using INSERT OR IGNORE to avoid duplicates
cur.execute("INSERT OR IGNORE INTO employees (name, age, department) VALUES ('Alice', 30, 'HR')")
cur.execute("INSERT OR IGNORE INTO employees (name, age, department) VALUES ('Bob', 24, 'Engineering')")
cur.execute("INSERT OR IGNORE INTO employees (name, age, department) VALUES ('Charlie', 28, 'Marketing')")

# Commit the changes
conn.commit()

db = SQLDatabase.from_uri("sqlite:///employees.db")

# Print the database dialect
print(db.dialect)

print(db.get_usable_table_names())

# Make the connection, cursor, and insert function available for import
def insert_employee(name: str, age: int, department: str):
    local_conn = sqlite3.connect('employees.db')
    local_cur = local_conn.cursor()
    local_cur.execute("INSERT OR IGNORE INTO employees (name, age, department) VALUES (?, ?, ?)", (name, age, department))
    local_conn.commit()
    local_conn.close()

