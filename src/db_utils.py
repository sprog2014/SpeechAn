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
    pg_pool = psycopg2.pool.ThreadedConnectionPool(
        minconn=1,
        maxconn=NUM_WORKERS + 5,
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
        pool_size=min(32, NUM_WORKERS + 5),
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

def insert_transcript(linkedid, channel, start, end, text, conn=None):
    def _execute(c):
        cur = c.cursor()
        cur.execute("""
            INSERT INTO transcripts (linkedid, channel, start_time, end_time, text)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (linkedid, channel, start, end, text))
        tid = cur.fetchone()[0]
        c.commit()
        return tid

    if conn:
        return _execute(conn)
    else:
        with get_pg_connection() as conn:
            return _execute(conn)

def insert_emotion(transcript_id, emotion, confidence, conn=None):
    def _execute(c):
        cur = c.cursor()
        cur.execute("""
            INSERT INTO speech_emotions (transcript_id, emotion, confidence)
            VALUES (%s, %s, %s)
        """, (transcript_id, emotion, confidence))
        c.commit()

    if conn:
        _execute(conn)
    else:
        with get_pg_connection() as conn:
            _execute(conn)

def get_default_prompt(conn=None):
    def _execute(c):
        cur = c.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT id, prompt_text FROM prompts WHERE is_default = TRUE LIMIT 1")
        return cur.fetchone()

    if conn:
        return _execute(conn)
    else:
        with get_pg_connection() as conn:
            return _execute(conn)

def get_prompt_by_id(prompt_id, conn=None):
    def _execute(c):
        cur = c.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT id, prompt_text FROM prompts WHERE id = %s", (prompt_id,))
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

def insert_evaluation(linkedid, prompt_id, result_json, conn=None):
    def _execute(c):
        cur = c.cursor()
        cur.execute("""
            INSERT INTO evaluations (linkedid, prompt_id, politeness_score, client_sentiment,
                                     call_purpose, call_summary, checklist_json, metrics_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (linkedid, prompt_id) DO UPDATE SET
                politeness_score = EXCLUDED.politeness_score,
                client_sentiment = EXCLUDED.client_sentiment,
                call_purpose = EXCLUDED.call_purpose,
                call_summary = EXCLUDED.call_summary,
                checklist_json = EXCLUDED.checklist_json,
                metrics_json = EXCLUDED.metrics_json
        """, (
            linkedid,
            prompt_id,
            result_json.get('politeness_score'),
            result_json.get('client_sentiment'),
            result_json.get('call_purpose'),
            result_json.get('call_summary'),
            json.dumps(result_json.get('checklist', {})),
            json.dumps(result_json.get('metrics', {}))
        ))
        c.commit()

    if conn:
        _execute(conn)
    else:
        with get_pg_connection() as conn:
            _execute(conn)
