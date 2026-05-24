import streamlit as st
import pandas as pd
import sys
import os
import subprocess
from datetime import datetime, timedelta

# Добавляем путь к src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from db_utils import (
    get_system_running_status, set_system_running_status,
    get_all_prompts, upsert_prompt, delete_prompt,
    get_pg_connection
)
from config import PG_CONFIG

if not st.session_state.get("password_correct", False):
    st.error("Пожалуйста, авторизуйтесь на главной странице.")
    st.stop()

st.title("Настройки и Управление")

# --- Раздел 1: Состояние системы ---
st.header("Состояние системы")
is_running = get_system_running_status()

col_status, col_btn = st.columns([3, 1])
if is_running:
    col_status.success("Система запущена и обрабатывает файлы.")
    if col_btn.button("Остановить"):
        set_system_running_status(False)
        st.rerun()
else:
    col_status.warning("Система находится в режиме ожидания (ОСТАНОВЛЕНА).")
    if col_btn.button("Запустить"):
        set_system_running_status(True)
        st.rerun()

# --- Раздел 2: Управление промптами ---
st.header("Управление промптами")
prompts = get_all_prompts()
df_prompts = pd.DataFrame(prompts)
st.dataframe(df_prompts)

with st.expander("Добавить / Редактировать промпт"):
    p_id = st.number_input("ID (оставьте 0 для нового)", min_value=0, value=0)
    p_name = st.text_input("Название")
    p_text = st.text_area("Текст промпта", height=200)
    p_default = st.checkbox("По умолчанию")

    if st.button("Сохранить промпт"):
        upsert_prompt(p_name, p_text, is_default=p_default, prompt_id=p_id if p_id > 0 else None)
        st.success("Сохранено!")
        st.rerun()

    if p_id > 0:
        if st.button("Удалить промпт", type="primary"):
            delete_prompt(p_id)
            st.success("Удалено!")
            st.rerun()

# --- Раздел 3: Статистика ---
st.header("Статистика обработки")

@st.cache_data(ttl=60)
def get_stats():
    with get_pg_connection() as conn:
        df_stats = pd.read_sql("""
            SELECT
                DATE(calldate) as date,
                COUNT(*) as count,
                AVG(processing_duration) as avg_duration,
                SUM(processing_duration) as total_duration
            FROM calls
            WHERE processing_status = 'done'
            GROUP BY DATE(calldate)
            ORDER BY DATE(calldate) DESC
        """, conn)

        df_prompts_stats = pd.read_sql("""
            SELECT p.name, COUNT(e.linkedid) as used_count
            FROM prompts p
            LEFT JOIN evaluations e ON p.id = e.prompt_id
            GROUP BY p.name
        """, conn)

    return df_stats, df_prompts_stats

df_stats, df_prompts_stats = get_stats()

st.subheader("По дням")
st.dataframe(df_stats)

st.subheader("Использование промптов")
st.dataframe(df_prompts_stats)

# --- Раздел 4: Ручной запуск ---
st.header("Ручной запуск")
with st.form("manual_run_form"):
    date_start = st.date_input("Дата начала", datetime.now() - timedelta(days=1))
    date_end = st.date_input("Дата конца", datetime.now())
    selected_prompt = st.selectbox("Промпт", options=[p['id'] for p in prompts], format_func=lambda x: next(p['name'] for p in prompts if p['id'] == x))

    submit_manual = st.form_submit_button("Запустить повторную обработку")

    if submit_manual:
        # Запуск manual_run.py в фоновом режиме
        cmd = [
            "python3", "src/manual_run.py",
            date_start.strftime("%Y-%m-%d"),
            date_end.strftime("%Y-%m-%d"),
            "--prompt_id", str(selected_prompt)
        ]
        # Используем Popen чтобы не блокировать веб-интерфейс
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        st.info(f"Процесс ручного запуска инициирован (PID: {process.pid}).")

# Кнопка перехода в дашборд (хотя он есть в боковой панели)
if st.button("Перейти к Аналитике"):
    st.switch_page("pages/1_analytics.py")
