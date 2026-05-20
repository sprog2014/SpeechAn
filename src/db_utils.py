import psycopg2
import mysql.connector
from config import PG_CONFIG, MYSQL_CONFIG
from contextlib import contextmanager

@contextmanager
def get_pg_connection():
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        yield conn
    finally:
        conn.close()

@contextmanager
def get_mysql_connection():
    conn = mysql.connector.connect(**MYSQL_CONFIG)
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
        return row

def upsert_call(metadata, file_path):
    with get_pg_connection() as conn:
        cur = conn.cursor()
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
        conn.commit()

def set_call_done(linkedid):
    with get_pg_connection() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE calls SET processing_status='done' WHERE linkedid=%s", (linkedid,))
        conn.commit()

def set_call_error(linkedid):
    with get_pg_connection() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE calls SET processing_status='error' WHERE linkedid=%s", (linkedid,))
        conn.commit()

def insert_transcript(linkedid, channel, start, end, text):
    with get_pg_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO transcripts (linkedid, channel, start_time, end_time, text)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (linkedid, channel, start, end, text))
        transcript_id = cur.fetchone()[0]
        conn.commit()
        return transcript_id

def insert_emotion(transcript_id, emotion, confidence):
    with get_pg_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO speech_emotions (transcript_id, emotion, confidence)
            VALUES (%s, %s, %s)
        """, (transcript_id, emotion, confidence))
        conn.commit()

def insert_evaluation(linkedid, result_json):
    with get_pg_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO evaluations (linkedid, politeness_score, client_sentiment,
                                     call_purpose, call_summary, checklist_json, metrics_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (linkedid) DO UPDATE SET
                politeness_score = EXCLUDED.politeness_score,
                client_sentiment = EXCLUDED.client_sentiment,
                call_purpose = EXCLUDED.call_purpose,
                call_summary = EXCLUDED.call_summary,
                checklist_json = EXCLUDED.checklist_json,
                metrics_json = EXCLUDED.metrics_json
        """, (
            linkedid,
            result_json.get('politeness_score'),
            result_json.get('client_sentiment'),
            result_json.get('call_purpose'),
            result_json.get('call_summary'),
            result_json.get('checklist', '{}'),
            result_json.get('metrics', '{}')
        ))
        conn.commit()