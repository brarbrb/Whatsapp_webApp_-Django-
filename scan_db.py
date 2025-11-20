import sqlite3
import sys
import os

def scan_sqlite(db_path):
    if not os.path.exists(db_path):
        print(f"Error: Database file not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    print(f"\n=== Scanning SQLite DB: {db_path} ===\n")

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;") # all tables
    tables = [row["name"] for row in cur.fetchall()]

    print("Tables found:")
    for t in tables:
        print("  -", t)

    print("\n")
    app_tables = [t for t in tables if t.startswith("whatsapp")]

    for table in app_tables:
        print("="*60)
        print(f"TABLE: {table}")
        print("="*60)
        # schema
        print("\nSchema:")
        cur.execute(f"PRAGMA table_info({table});")
        columns = cur.fetchall()
        for col in columns:
            print(f"  {col['name']} ({col['type']})")
        # row count
        cur.execute(f"SELECT COUNT(*) FROM {table};")
        count = cur.fetchone()[0]
        print(f"\nRow count: {count}")
        # sample records
        if count > 0:
            print("\nFirst 5 rows:")
            cur.execute(f"SELECT * FROM {table} LIMIT 5;")
            rows = cur.fetchall()

            for r in rows:
                print(dict(r))
        else:
            print("No rows.\n")
        print("\n")
    conn.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scan_db.py <path_to_db.sqlite3>")
    else:
        scan_sqlite(sys.argv[1])
# python scan_db.py db.sqlite3