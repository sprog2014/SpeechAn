import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
import mysql.connector
from mysql.connector import pooling
from config import PG_CONFIG, MYSQL_CONFIG, NUM_WORKERS
from contextlib import contextmanager
import logging
import json

# Инициализация пула соединений PostgreSQL
try:
    # Используем SimpleConnectionPool для процессов воркеров,
    # так как каждый воркер - это отдельный процесс и не требует потокобезопасного пула внутри себя.
    # Но ThreadedConnectionPool тоже подходит.
    pg_pool = psycopg2.pool.ThreadedConnectionPool(
        minconn=1,
        maxconn=5, # Ограничиваем количество соединений на один процесс
        **PG_CONFIG
    )
    logging.info("PostgreSQL connection pool created")
except Exception as e:
    logging.error(f"Error creating PostgreSQL pool: {e}")
    pg_pool = None

# Инициализация пула соединений MySQL
try:
    mysql_pool = mysql.connector.pooling.MySQLConnectionPool(
        pool_name="mypool",
        pool_size=5, # Ограничиваем количество соединений на один процесс
        **MYSQL_CONFIG
    )
    logging.info("MySQL connection pool created")
except Exception as e:
    logging.error(f"Error creating MySQL pool: {e}")
    mysql_pool = None

@contextmanager
def get_pg_connection():
    if pg_pool is None:
        conn = psycopg2.connect(**PG_CONFIG)
        try:
            yield conn
        finally:
            conn.close()
    else:
        conn = pg_pool.getconn()
        try:
            yield conn
        finally:
            pg_pool.putconn(conn)

@contextmanager
def get_mysql_connection():
    if mysql_pool is None:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        try:
            yield conn
        finally:
            conn.close()
    else:
        conn = mysql_pool.get_connection()
        try:
            yield conn
        finally:
            conn.close()

def fetch_call_metadata(linkedid):
    with get_mysql_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT calldate, src, answeredext, direction, duration, billsec,
                   fromtrunksrc, moduleparams, incomingTrunk
            FROM analytics
            WHERE linkedid = %s AND modulename = 'call'
            LIMIT 1
        """, (linkedid,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"No metadata for linkedid {linkedid}")
        row['linkedid'] = linkedid
        return row

def upsert_call(metadata, file_path, conn=None):
    def _execute(c):
        cur = c.cursor()
        cur.execute("""
            INSERT INTO calls (linkedid, calldate, src, answeredext, direction,
                               duration, billsec, fromtrunksrc, moduleparams,
                               incomingtrunk, file_path, processing_status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'processing')
            ON CONFLICT (linkedid) DO UPDATE SET processing_status = 'processing'
        """, (
            metadata['linkedid'],
            metadata['calldate'],
            metadata['src'],
            metadata['answeredext'],
            metadata['direction'],
            metadata['duration'],
            metadata['billsec'],
            metadata['fromtrunksrc'],
            metadata['moduleparams'],
            metadata['incomingTrunk'],
            file_path
        ))
        c.commit()

    if conn:
        _execute(conn)
    else:
        with get_pg_connection() as conn:
            _execute(conn)

def get_aggregated_asr_metrics(linkedid, conn=None):
    """
    Вычисляет среднюю четкость и темп речи для оператора и клиента на основе данных из таблицы транскрибации.
    """
    def _execute(c):
        cur = c.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT
                channel,
                AVG(diction) as avg_diction,
                AVG(wpm) as avg_wpm
            FROM transcripts
            WHERE linkedid = %s AND diction IS NOT NULL AND wpm IS NOT NULL
            GROUP BY channel
        """, (linkedid,))
        rows = cur.fetchall()

        metrics = {
            "operator_diction": 0.0,
            "operator_wpm": 0,
            "client_diction": 0.0,
            "client_wpm": 0
        }

        for row in rows:
            if row['channel'] == 'operator':
                metrics["operator_diction"] = round(float(row['avg_diction']), 1) if row['avg_diction'] else 0.0
                metrics["operator_wpm"] = int(row['avg_wpm']) if row['avg_wpm'] else 0
            elif row['channel'] == 'client':
                metrics["client_diction"] = round(float(row['avg_diction']), 1) if row['avg_diction'] else 0.0
                metrics["client_wpm"] = int(row['avg_wpm']) if row['avg_wpm'] else 0
        return metrics

    if conn:
        return _execute(conn)
    else:
        with get_pg_connection() as conn:
            return _execute(conn)

def get_all_tasks(conn=None):
    def _execute(c):
        cur = c.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT t.*, p.name as prompt_name
            FROM tasks t
            JOIN prompts p ON t.prompt_id = p.id
            ORDER BY t.created_at DESC
        """)
        return cur.fetchall()

    if conn:
        return _execute(conn)
    else:
        with get_pg_connection() as conn:
            return _execute(conn)

def get_call_status(linkedid, conn=None):
    def _execute(c):
        cur = c.cursor()
        cur.execute("SELECT processing_status FROM calls WHERE linkedid = %s", (linkedid,))
        row = cur.fetchone()
        return row[0] if row else None

    if conn:
        return _execute(conn)
    else:
        with get_pg_connection() as conn:
            return _execute(conn)

def add_task(prompt_id, start_date, end_date, analyze_all, conn=None):
    def _execute(c):
        cur = c.cursor()
        cur.execute("""
            INSERT INTO tasks (prompt_id, start_date, end_date, analyze_all)
            VALUES (%s, %s, %s, %s)
        """, (prompt_id, start_date, end_date, analyze_all))
        c.commit()

    if conn:
        _execute(conn)
    else:
        with get_pg_connection() as conn:
            _execute(conn)

def delete_task(task_id, conn=None):
    def _execute(c):
        cur = c.cursor()
        cur.execute("DELETE FROM tasks WHERE id = %s", (task_id,))
        c.commit()

    if conn:
        _execute(conn)
    else:
        with get_pg_connection() as conn:
            _execute(conn)

def update_task_status(task_id, asr_status=None, llm_status=None, conn=None):
    def _execute(c):
        cur = c.cursor()
        if asr_status:
            cur.execute("UPDATE tasks SET asr_status = %s WHERE id = %s", (asr_status, task_id))
        if llm_status:
            cur.execute("UPDATE tasks SET llm_status = %s WHERE id = %s", (llm_status, task_id))
        c.commit()

    if conn:
        _execute(conn)
    else:
        with get_pg_connection() as conn:
            _execute(conn)

def get_active_tasks(conn=None):
    def _execute(c):
        cur = c.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT * FROM tasks
            WHERE asr_status != 'completed' OR llm_status != 'completed'
            ORDER BY created_at ASC
        """)
        return cur.fetchall()

    if conn:
        return _execute(conn)
    else:
        with get_pg_connection() as conn:
            return _execute(conn)

def insert_processing_stats(linkedid, asr_dur, llm_dur, total_dur, conn=None):
    def _execute(c):
        cur = c.cursor()
        cur.execute("""
            INSERT INTO processing_stats (linkedid, asr_duration, llm_duration, total_duration)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (linkedid) DO UPDATE SET
                asr_duration = EXCLUDED.asr_duration,
                llm_duration = EXCLUDED.llm_duration,
                total_duration = EXCLUDED.total_duration
        """, (linkedid, asr_dur, llm_dur, total_dur))
        c.commit()

    if conn:
        _execute(conn)
    else:
        with get_pg_connection() as conn:
            _execute(conn)

from datetime import timedelta

def get_processing_statistics(start_date, end_date):
    # Корректируем end_date, чтобы захватить весь день
    actual_end = end_date + timedelta(days=1)

    with get_pg_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # 1. Сводка по дням: пропущено, обработано, в процессе, ошибка + времена
        cur.execute("""
            SELECT
                DATE(calldate) as date,
                COUNT(*) FILTER (WHERE processing_status = 'skipped') as skipped,
                COUNT(*) FILTER (WHERE processing_status = 'done') as processed,
                COUNT(*) FILTER (WHERE processing_status = 'processing') as in_progress,
                COUNT(*) FILTER (WHERE processing_status = 'transcribed') as transcribed,
                COUNT(*) FILTER (WHERE processing_status = 'error') as error,
                COUNT(*) FILTER (WHERE processing_status = 'empty') as empty,
                COUNT(*) FILTER (WHERE processing_status = 'stop') as stop,
                ROUND(AVG(processing_duration) FILTER (WHERE processing_status = 'done')::numeric, 2) as avg_duration,
                ROUND(SUM(processing_duration) FILTER (WHERE processing_status = 'done')::numeric, 2) as total_duration
            FROM calls
            WHERE calldate >= %s AND calldate < %s
            GROUP BY DATE(calldate)
            ORDER BY DATE(calldate) DESC
        """, (start_date, actual_end))
        daily_stats = cur.fetchall()

        # 2. Скорость анализа: файлов в час
        # Берем данные из processing_stats
        cur.execute("""
            SELECT
                DATE(created_at) as date,
                COUNT(*) as count,
                AVG(total_duration) as avg_total_duration
            FROM processing_stats
            WHERE created_at >= %s AND created_at < %s
            GROUP BY DATE(created_at)
        """, (start_date, actual_end))
        speed_stats = cur.fetchall()

        # 3. Распределение по этапам
        cur.execute("""
            SELECT
                AVG(asr_duration) as avg_asr,
                AVG(llm_duration) as avg_llm,
                AVG(total_duration) as avg_total
            FROM processing_stats
            WHERE created_at >= %s AND created_at < %s
        """, (start_date, actual_end))
        timings = cur.fetchone()

        return {
            "daily_stats": daily_stats,
            "speed_stats": speed_stats,
            "timings": timings
        }

def get_prompt_usage_statistics():
    with get_pg_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT p.name as "name", COUNT(e.linkedid) as "count"
            FROM prompts p
            LEFT JOIN evaluations e ON p.id = e.prompt_id
            GROUP BY p.name
            ORDER BY "count" DESC
        """)
        return cur.fetchall()

def get_all_phones(conn=None):
    def _execute(c):
        cur = c.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT number, name, use FROM phones ORDER BY name ASC")
        return cur.fetchall()

    if conn:
        return _execute(conn)
    else:
        with get_pg_connection() as conn:
            return _execute(conn)

def update_phone_use(number, use, conn=None):
    def _execute(c):
        cur = c.cursor()
        cur.execute("UPDATE phones SET use = %s WHERE number = %s", (use, number))
        c.commit()

    if conn:
        _execute(conn)
    else:
        with get_pg_connection() as conn:
            _execute(conn)

def check_phone_usage(number, conn=None):
    if not number:
        return False
    def _execute(c):
        cur = c.cursor()
        cur.execute("SELECT use FROM phones WHERE number = %s", (number,))
        row = cur.fetchone()
        return row[0] if row else False

    if conn:
        return _execute(conn)
    else:
        with get_pg_connection() as conn:
            return _execute(conn)

def sync_phones_from_external_db(conn=None):
    # 1. Получаем текущие настройки из PostgreSQL
    current_phones = {}
    phones_data = get_all_phones(conn=conn)
    for p in phones_data:
        current_phones[p['number']] = p['use']

    # 2. Получаем список из MySQL bitpbx.users
    new_phones = []
    with get_mysql_connection() as m_conn:
        m_cur = m_conn.cursor(dictionary=True)
        m_cur.execute("SELECT name, number FROM bitpbx.users")
        new_phones = m_cur.fetchall()

    # 3. Обновляем таблицу phones в PostgreSQL
    def _execute(c):
        cur = c.cursor()
        cur.execute("DELETE FROM phones")
        for p in new_phones:
            number = p['number']
            name = p['name']
            # Сохраняем старое значение use, если оно было
            use = current_phones.get(number, True)
            cur.execute("""
                INSERT INTO phones (number, name, use)
                VALUES (%s, %s, %s)
                ON CONFLICT (number) DO UPDATE SET name = EXCLUDED.name, use = EXCLUDED.use
            """, (number, name, use))
        c.commit()

    if conn:
        _execute(conn)
    else:
        with get_pg_connection() as conn:
            _execute(conn)

def set_default_prompt(prompt_id, conn=None):
    def _execute(c):
        cur = c.cursor()
        cur.execute("UPDATE prompts SET is_default = FALSE WHERE is_default = TRUE")
        cur.execute("UPDATE prompts SET is_default = TRUE WHERE id = %s", (prompt_id,))
        c.commit()

    if conn:
        _execute(conn)
    else:
        with get_pg_connection() as conn:
            _execute(conn)

def set_processing_duration(linkedid, duration, conn=None):
    def _execute(c):
        cur = c.cursor()
        cur.execute("UPDATE calls SET processing_duration=%s WHERE linkedid=%s", (duration, linkedid))
        c.commit()

    if conn:
        _execute(conn)
    else:
        with get_pg_connection() as conn:
            _execute(conn)

def get_system_setting(key, default='false', conn=None):
    def _execute(c):
        cur = c.cursor()
        cur.execute("SELECT value FROM system_settings WHERE key = %s", (key,))
        row = cur.fetchone()
        return row[0] if row else default

    if conn:
        return _execute(conn)
    else:
        with get_pg_connection() as conn:
            return _execute(conn)

def set_system_setting(key, value, conn=None):
    def _execute(c):
        cur = c.cursor()
        cur.execute("""
            INSERT INTO system_settings (key, value)
            VALUES (%s, %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        """, (key, str(value).lower()))
        c.commit()

    if conn:
        _execute(conn)
    else:
        with get_pg_connection() as conn:
            _execute(conn)

def get_system_running_status(conn=None):
    val = get_system_setting('is_running', default='true', conn=conn)
    return val.lower() == 'true'

def set_system_running_status(is_running, conn=None):
    set_system_setting('is_running', is_running, conn=conn)

def get_all_prompts(conn=None):
    def _execute(c):
        cur = c.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT id, name, prompt_text, is_default, schema_json, created_at FROM prompts ORDER BY id ASC")
        return cur.fetchall()

    if conn:
        return _execute(conn)
    else:
        with get_pg_connection() as conn:
            return _execute(conn)

def is_phone_registered(number, conn=None):
    if not number:
        return False
    def _execute(c):
        cur = c.cursor()
        cur.execute("SELECT 1 FROM phones WHERE number = %s", (number,))
        return cur.fetchone() is not None

    if conn:
        return _execute(conn)
    else:
        with get_pg_connection() as conn:
            return _execute(conn)

def upsert_prompt(name, prompt_text, is_default=False, prompt_id=None, schema_json=None, conn=None):
    def _execute(c):
        cur = c.cursor()
        schema_val = schema_json if schema_json else '[]'
        if isinstance(schema_val, list) or isinstance(schema_val, dict):
            schema_val = json.dumps(schema_val, ensure_ascii=False)
        if prompt_id:
            if is_default:
                cur.execute("UPDATE prompts SET is_default = FALSE WHERE is_default = TRUE")
            cur.execute("""
                UPDATE prompts SET name=%s, prompt_text=%s, is_default=%s, schema_json=%s
                WHERE id=%s
            """, (name, prompt_text, is_default, schema_val, prompt_id))
        else:
            if is_default:
                cur.execute("UPDATE prompts SET is_default = FALSE WHERE is_default = TRUE")
            cur.execute("""
                INSERT INTO prompts (name, prompt_text, is_default, schema_json)
                VALUES (%s, %s, %s, %s)
            """, (name, prompt_text, is_default, schema_val))
        c.commit()

    if conn:
        _execute(conn)
    else:
        with get_pg_connection() as conn:
            _execute(conn)

def delete_prompt(prompt_id, conn=None):
    def _execute(c):
        cur = c.cursor()
        cur.execute("DELETE FROM prompts WHERE id=%s", (prompt_id,))
        c.commit()

    if conn:
        _execute(conn)
    else:
        with get_pg_connection() as conn:
            _execute(conn)

def get_value_mappings(conn=None):
    def _execute(c):
        cur = c.cursor()
        cur.execute("SELECT value FROM system_settings WHERE key = 'value_mappings'")
        row = cur.fetchone()
        if row:
            try:
                return json.loads(row[0])
            except:
                return []
        return []

    if conn:
        return _execute(conn)
    else:
        with get_pg_connection() as conn:
            return _execute(conn)

def set_value_mappings(mappings, conn=None):
    def _execute(c):
        cur = c.cursor()
        cur.execute("""
            INSERT INTO system_settings (key, value)
            VALUES ('value_mappings', %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        """, (json.dumps(mappings, ensure_ascii=False),))
        c.commit()

    if conn:
        _execute(conn)
    else:
        with get_pg_connection() as conn:
            _execute(conn)

def update_evaluations_value(prompt_id, column, old_value, new_value, conn=None):
    if column not in ['call_purpose', 'client_sentiment']:
        return
    def _execute(c):
        cur = c.cursor()
        query = f"UPDATE evaluations SET {column} = %s WHERE prompt_id = %s AND {column} = %s"
        cur.execute(query, (new_value, prompt_id, old_value))
        c.commit()

    if conn:
        _execute(conn)
    else:
        with get_pg_connection() as conn:
            _execute(conn)

def get_field_synonyms(prompt_id, conn=None):
    def _execute(c):
        cur = c.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT technical_name, synonym FROM field_synonyms WHERE prompt_id = %s", (prompt_id,))
        rows = cur.fetchall()
        return {r['technical_name']: r['synonym'] for r in rows}

    if conn:
        return _execute(conn)
    else:
        with get_pg_connection() as conn:
            return _execute(conn)

def set_field_synonyms(prompt_id, synonyms_dict, conn=None):
    def _execute(c):
        cur = c.cursor()
        # Удаляем старые и вставляем новые
        cur.execute("DELETE FROM field_synonyms WHERE prompt_id = %s", (prompt_id,))
        for tech_name, syn in synonyms_dict.items():
            if syn and str(syn).strip():
                cur.execute("""
                    INSERT INTO field_synonyms (prompt_id, technical_name, synonym)
                    VALUES (%s, %s, %s)
                """, (prompt_id, tech_name, syn))
        c.commit()

    if conn:
        _execute(conn)
    else:
        with get_pg_connection() as conn:
            _execute(conn)

def set_call_status(linkedid, status, conn=None):
    def _execute(c):
        cur = c.cursor()
        cur.execute("UPDATE calls SET processing_status=%s WHERE linkedid=%s", (status, linkedid))
        c.commit()

    if conn:
        _execute(conn)
    else:
        with get_pg_connection() as conn:
            _execute(conn)

def set_call_done(linkedid, conn=None):
    set_call_status(linkedid, 'done', conn)

def set_call_error(linkedid, conn=None):
    set_call_status(linkedid, 'error', conn)

def insert_transcript(linkedid, channel, start, end, text, diction=None, wpm=None, conn=None):
    def _execute(c):
        cur = c.cursor()
        cur.execute("""
            INSERT INTO transcripts (linkedid, channel, start_time, end_time, text, diction, wpm)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (linkedid, channel, start, end, text, diction, wpm))
        tid = cur.fetchone()[0]
        c.commit()
        return tid

    if conn:
        return _execute(conn)
    else:
        with get_pg_connection() as conn:
            return _execute(conn)

def get_default_prompt(conn=None):
    def _execute(c):
        cur = c.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT id, prompt_text, schema_json FROM prompts WHERE is_default = TRUE LIMIT 1")
        return cur.fetchone()

    if conn:
        return _execute(conn)
    else:
        with get_pg_connection() as conn:
            return _execute(conn)

def get_prompt_by_id(prompt_id, conn=None):
    def _execute(c):
        cur = c.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT id, prompt_text, schema_json FROM prompts WHERE id = %s", (prompt_id,))
        return cur.fetchone()

    if conn:
        return _execute(conn)
    else:
        with get_pg_connection() as conn:
            return _execute(conn)

def check_transcript_exists(linkedid, conn=None):
    def _execute(c):
        cur = c.cursor()
        cur.execute("SELECT 1 FROM transcripts WHERE linkedid = %s LIMIT 1", (linkedid,))
        return cur.fetchone() is not None

    if conn:
        return _execute(conn)
    else:
        with get_pg_connection() as conn:
            return _execute(conn)

def check_evaluation_exists(linkedid, prompt_id, conn=None):
    def _execute(c):
        cur = c.cursor()
        cur.execute("SELECT 1 FROM evaluations WHERE linkedid = %s AND prompt_id = %s LIMIT 1", (linkedid, prompt_id))
        return cur.fetchone() is not None

    if conn:
        return _execute(conn)
    else:
        with get_pg_connection() as conn:
            return _execute(conn)

def get_call_file_path(linkedid, conn=None):
    def _execute(c):
        cur = c.cursor()
        cur.execute("SELECT file_path FROM calls WHERE linkedid = %s", (linkedid,))
        row = cur.fetchone()
        return row[0] if row else None

    if conn:
        return _execute(conn)
    else:
        with get_pg_connection() as conn:
            return _execute(conn)

def get_call_transcript(linkedid, conn=None):
    def _execute(c):
        cur = c.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT channel, start_time, text
            FROM transcripts
            WHERE linkedid = %s
            ORDER BY start_time ASC
        """, (linkedid,))
        return cur.fetchall()

    if conn:
        return _execute(conn)
    else:
        with get_pg_connection() as conn:
            return _execute(conn)

def format_dialogue(transcript_rows):
    """
    Централизованная сборка текста диалога с русскими метками ролей.
    Используется как для анализа, так и для вывода на экран.
    """
    formatted_lines = []
    for row in transcript_rows:
        label = "Оператор" if row['channel'] == 'operator' else "Клиент"
        formatted_lines.append(f"{label}: {row['text']}")
    return "\n".join(formatted_lines)

def build_case_sql(column, mapping_list, default_label):
    """
    Генерирует SQL CASE для маппинга технических значений в человекочитаемые метки.
    """
    if not mapping_list:
        default_label_esc = str(default_label).replace("'", "''")
        return f"COALESCE(e.{column}, '{default_label_esc}')"

    sql = f"CASE e.{column} "
    for item in mapping_list:
        for k, v in item.items():
            # Экранируем одинарные кавычки для безопасности SQL
            k_esc = str(k).replace("'", "''")
            v_esc = str(v).replace("'", "''")
            sql += f"WHEN '{k_esc}' THEN '{v_esc}' "

    default_label_esc = str(default_label).replace("'", "''")
    sql += f"ELSE COALESCE(e.{column}, '{default_label_esc}') END"
    return sql

def get_rating_from_mysql(linkedid):
    """
    Получает оценку (rating) из таблицы bitpbx.evaluation_reports по linkedid.
    Если ничего не найдено или произошла ошибка, возвращает 0.
    """
    try:
        with get_mysql_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT value
                FROM bitpbx.evaluation_reports
                WHERE linkedid = %s
                LIMIT 1
            """, (linkedid,))
            row = cursor.fetchone()
            if row and row[0] is not None:
                return int(row[0])
            return 0
    except Exception as e:
        logging.error(f"Error fetching rating for {linkedid}: {e}")
        return 0

def insert_evaluation(linkedid, prompt_id, result_json, rating=0, conn=None):
    def _execute(c):
        cur = c.cursor()
        cur.execute("""
            INSERT INTO evaluations (linkedid, prompt_id, politeness_score, client_sentiment,
                                     call_purpose, call_summary, checklist_json, metrics_json, rating)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (linkedid, prompt_id) DO UPDATE SET
                politeness_score = EXCLUDED.politeness_score,
                client_sentiment = EXCLUDED.client_sentiment,
                call_purpose = EXCLUDED.call_purpose,
                call_summary = EXCLUDED.call_summary,
                checklist_json = EXCLUDED.checklist_json,
                metrics_json = EXCLUDED.metrics_json,
                rating = EXCLUDED.rating
        """, (
            linkedid,
            prompt_id,
            result_json.get('politeness_score'),
            result_json.get('client_sentiment'),
            result_json.get('call_purpose'),
            result_json.get('call_summary'),
            json.dumps(result_json.get('checklist', {})),
            json.dumps(result_json.get('metrics', {})),
            rating
        ))
        c.commit()

    if conn:
        _execute(conn)
    else:
        with get_pg_connection() as conn:
            _execute(conn)
