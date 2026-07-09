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

        print("Updating processing_status check constraint to include 'transcribed', 'empty' and 'stop'...")
        cur.execute("ALTER TABLE calls DROP CONSTRAINT IF EXISTS calls_processing_status_check;")
        cur.execute("""
            ALTER TABLE calls ADD CONSTRAINT calls_processing_status_check
            CHECK (processing_status IN ('new','processing','transcribed','done','error','skipped','empty','stop'));
        """)

        print("Creating tasks table if not exists...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id              SERIAL PRIMARY KEY,
                prompt_id       INT NOT NULL REFERENCES prompts(id) ON DELETE CASCADE,
                start_date      DATE NOT NULL,
                end_date        DATE NOT NULL,
                analyze_all     BOOLEAN DEFAULT FALSE,
                asr_status      VARCHAR(20) DEFAULT 'planned' CHECK (asr_status IN ('planned','processing','completed')),
                llm_status      VARCHAR(20) DEFAULT 'planned' CHECK (llm_status IN ('planned','processing','completed')),
                created_at      TIMESTAMP DEFAULT now()
            );
        """)

        print("Creating reports table if not exists...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id              SERIAL PRIMARY KEY,
                name            VARCHAR(200) NOT NULL UNIQUE,
                settings        JSONB NOT NULL,
                created_at      TIMESTAMP DEFAULT now()
            );
        """)

        print("Creating field_synonyms table if not exists...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS field_synonyms (
                prompt_id INT NOT NULL REFERENCES prompts(id) ON DELETE CASCADE,
                technical_name VARCHAR(100) NOT NULL,
                synonym VARCHAR(255) NOT NULL,
                PRIMARY KEY (prompt_id, technical_name)
            );
        """)

        print("Creating JSONB GIN indexes if they do not exist...")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_evals_checklist_gin ON evaluations USING GIN (checklist_json);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_evals_metrics_gin ON evaluations USING GIN (metrics_json);")

        print("Adding diction and wpm columns to transcripts if they do not exist...")
        cur.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='transcripts' AND column_name='diction') THEN
                    ALTER TABLE transcripts ADD COLUMN diction NUMERIC(5,2);
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='transcripts' AND column_name='wpm') THEN
                    ALTER TABLE transcripts ADD COLUMN wpm INT;
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='evaluations' AND column_name='rating') THEN
                    ALTER TABLE evaluations ADD COLUMN rating SMALLINT DEFAULT 0;
                END IF;
            END $$;
        """)

        conn.commit()
        cur.close()
        conn.close()
        print("Schema updates applied successfully.")
    except Exception as e:
        print(f"Error applying schema updates: {e}")

if __name__ == "__main__":
    apply_schema_updates()
