import streamlit as st
import pandas as pd
import psycopg2
import sys
import os

# Добавляем путь к src, чтобы найти config.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from config import PG_CONFIG

st.set_page_config(page_title="Call Analysis Dashboard")
st.title("Аналитика звонков")

@st.cache_data(ttl=60)
def get_summary_data():
    conn = psycopg2.connect(**PG_CONFIG)
    df = pd.read_sql("""
        SELECT c.linkedid, c.calldate, c.direction, c.billsec, c.moduleparams,
               e.politeness_score, e.client_sentiment, e.call_purpose,
               e.checklist_json->>'greeting' as greeting,
               e.checklist_json->>'farewell' as farewell
        FROM calls c
        LEFT JOIN evaluations e ON c.linkedid = e.linkedid
        WHERE c.processing_status = 'done'
        ORDER BY c.calldate DESC
        LIMIT 500
    """, conn)
    conn.close()
    return df

df = get_summary_data()
st.dataframe(df)

st.subheader("Средняя вежливость")
avg_politeness = df['politeness_score'].mean()
st.metric("Средняя вежливость", f"{avg_politeness:.2f}")

st.subheader("Распределение целей звонков")
purpose_counts = df['call_purpose'].value_counts()
st.bar_chart(purpose_counts)
