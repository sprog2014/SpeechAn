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
    get_all_prompts, upsert_prompt, delete_prompt, set_default_prompt,
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

@st.dialog("Редактирование промпта")
def edit_prompt_dialog(prompt=None):
    p_id = prompt['id'] if prompt else None
    name = st.text_input("Название", value=prompt['name'] if prompt else "")
    text = st.text_area("Текст промпта", value=prompt['prompt_text'] if prompt else "", height=300)

    if st.button("Сохранить"):
        if name and text:
            upsert_prompt(name, text, is_default=prompt['is_default'] if prompt else False, prompt_id=p_id)
            st.success("Сохранено!")
            st.rerun()
        else:
            st.error("Название и текст не могут быть пустыми.")

prompts = get_all_prompts()
if prompts:
    # Подготовка данных для таблицы
    # "Текст в строке is_default=true показывай жирным"
    # В Streamlit dataframe/table это можно сделать через Style или просто добавив markdown,
    # но st.dataframe плохо поддерживает markdown внутри ячеек для отображения.
    # Используем column_config или просто преобразуем данные.

    display_data = []
    for p in prompts:
        name_display = f"**{p['name']}**" if p['is_default'] else p['name']
        display_data.append({
            "ID": p['id'], # Скрытый или для справки
            "Создан": p['created_at'].strftime("%Y-%m-%d %H:%M") if p['created_at'] else "",
            "Название": name_display,
            "_raw": p # Сохраняем оригинал для кнопок
        })

    df_display = pd.DataFrame(display_data)

    # "Строки таблицы должны быть выделяемыми без множественного выделения"
    selection = st.dataframe(
        df_display[["Создан", "Название"]],
        on_select="rerun",
        selection_mode="single_row",
        use_container_width=True,
        hide_index=True
    )

    selected_indices = selection.get("selection", {}).get("rows", [])
    selected_prompt = None
    if selected_indices:
        selected_prompt = display_data[selected_indices[0]]["_raw"]

    col1, col2, col3 = st.columns(3)

    if col1.button("Добавить", use_container_width=True):
        edit_prompt_dialog()

    if col2.button("Изменить", use_container_width=True, disabled=selected_prompt is None):
        edit_prompt_dialog(selected_prompt)

    if col3.button("По умолчанию", use_container_width=True, disabled=selected_prompt is None):
        set_default_prompt(selected_prompt['id'])
        st.success(f"Промпт '{selected_prompt['name']}' установлен по умолчанию.")
        st.rerun()

    if selected_prompt:
        if st.button("Удалить выбранный промпт", type="primary"):
            delete_prompt(selected_prompt['id'])
            st.success("Удалено!")
            st.rerun()
else:
    st.info("Промпты не найдены.")
    if st.button("Добавить первый промпт"):
        edit_prompt_dialog()

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
st.dataframe(df_stats, use_container_width=True)

st.subheader("Использование промптов")
st.dataframe(df_prompts_stats, use_container_width=True)

# --- Раздел 4: Ручной запуск ---
st.header("Ручной запуск")
with st.form("manual_run_form"):
    date_start = st.date_input("Дата начала", datetime.now() - timedelta(days=1))
    date_end = st.date_input("Дата конца", datetime.now())

    prompt_options = {p['id']: p['name'] for p in prompts}
    selected_manual_prompt = st.selectbox("Промпт", options=list(prompt_options.keys()), format_func=lambda x: prompt_options[x])

    submit_manual = st.form_submit_button("Запустить повторную обработку")

    if submit_manual:
        cmd = [
            "python3", "src/manual_run.py",
            date_start.strftime("%Y-%m-%d"),
            date_end.strftime("%Y-%m-%d"),
            "--prompt_id", str(selected_manual_prompt)
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        st.info(f"Процесс ручного запуска инициирован (PID: {process.pid}).")

if st.button("Перейти к Аналитике"):
    st.switch_page("pages/1_analytics.py")
