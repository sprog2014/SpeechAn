import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from db_utils import get_pg_connection

def update_schema():
    with get_pg_connection() as conn:
        cur = conn.cursor()

        print("Adding processing_duration to calls table...")
        try:
            cur.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS processing_duration REAL;")
            conn.commit()
            print("Done.")
        except Exception as e:
            conn.rollback()
            print(f"Error adding column: {e}")

        print("Creating system_settings table...")
        try:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS system_settings (
                    key VARCHAR(50) PRIMARY KEY,
                    value TEXT
                );
            """)
            # Initialize is_running if not exists
            cur.execute("""
                INSERT INTO system_settings (key, value)
                VALUES ('is_running', 'true')
                ON CONFLICT (key) DO NOTHING;
            """)
            conn.commit()
            print("Done.")
        except Exception as e:
            conn.rollback()
            print(f"Error creating table: {e}")

if __name__ == "__main__":
    update_schema()
