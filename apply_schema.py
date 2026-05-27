import psycopg2
import os
from pathlib import Path

# Пытаемся загрузить .env из /opt/calls/.env или из текущей папки
try:
    from dotenv import load_dotenv
    # Сначала проверяем глобальный путь
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

def apply_schema_updates():
    try:
        conn = psycopg2.connect(**PG_CONFIG)
        cur = conn.cursor()

        print("Updating processing_status check constraint...")
        # Drop old constraint
        cur.execute("""
            ALTER TABLE calls DROP CONSTRAINT IF EXISTS calls_processing_status_check;
        """)
        # Add new constraint
        cur.execute("""
            ALTER TABLE calls ADD CONSTRAINT calls_processing_status_check
            CHECK (processing_status IN ('new','processing','done','error','skipped'));
        """)

        print("Creating processing_stats table...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS processing_stats (
                linkedid        VARCHAR(32) PRIMARY KEY REFERENCES calls(linkedid) ON DELETE CASCADE,
                asr_duration    REAL,
                emotion_duration REAL,
                llm_duration    REAL,
                total_duration  REAL,
                created_at      TIMESTAMP DEFAULT now()
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_stats_created_at ON processing_stats(created_at);")

        conn.commit()
        cur.close()
        conn.close()
        print("Schema updates applied successfully.")
    except Exception as e:
        print(f"Error applying schema updates: {e}")

if __name__ == "__main__":
    apply_schema_updates()
