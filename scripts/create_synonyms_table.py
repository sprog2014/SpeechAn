import psycopg2
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    global_env = Path("/opt/calls/.env")
    if global_env.exists():
        load_dotenv(dotenv_path=global_env)
    else:
        load_dotenv()
except ImportError:
    pass

PG_CONFIG = {
    "host": os.getenv("PG_HOST"),
    "port": int(os.getenv("PG_PORT", 5432)),
    "dbname": os.getenv("PG_DB"),
    "user": os.getenv("PG_USER"),
    "password": os.getenv("PG_PASSWORD")
}

def create_synonyms_table():
    try:
        conn = psycopg2.connect(**PG_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS field_synonyms (
                prompt_id INT NOT NULL REFERENCES prompts(id) ON DELETE CASCADE,
                technical_name VARCHAR(100) NOT NULL,
                synonym VARCHAR(255) NOT NULL,
                PRIMARY KEY (prompt_id, technical_name)
            );
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("Table field_synonyms created or already exists.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    create_synonyms_table()
